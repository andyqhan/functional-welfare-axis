"""SFT training for maze navigation agents.

Uses trl.SFTTrainer with LoRA for supervised fine-tuning on
expert (perfect) maze trajectories.
"""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer

from src.maze.game import DIRECTIONS, Game
from src.maze.inference import compute_reward, parse_direction
from src.maze.maze import MazeGenerator
from src.maze.tiles import TileConfig
from src.sft_trainer.config import SFTConfig, parse_args
from src.sft_trainer.dataset import load_or_generate_dataset


# --- Checkpoint callback ---


class LoRACheckpointCallback(TrainerCallback):
    """Save checkpoints in the concept-vector-pipeline-compatible layout.

    - LoRA: global_step_N/lora_adapter/ containing adapter_{config,model}.*
    - FFT:  global_step_N/ is a symlink to the HF checkpoint-N dir (which
      already contains config.json + model.safetensors{,.index.json} +
      tokenizer files). The optimizer/scheduler state sits in the same dir
      but downstream loaders ignore it.
    """

    def __init__(self, config: SFTConfig):
        self.config = config

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        output_dir = Path(args.output_dir)
        hf_ckpt = output_dir / f"checkpoint-{step}"
        global_step_dir = output_dir / f"global_step_{step}"

        if self.config.use_lora:
            compat_dir = global_step_dir / "lora_adapter"
            compat_dir.mkdir(parents=True, exist_ok=True)
            for f in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
                src = hf_ckpt / f
                if src.exists():
                    shutil.copy2(src, compat_dir / f)
        else:
            # FFT: point global_step_N at the HF checkpoint dir via symlink so
            # we don't duplicate ~8GB of full-model weights per step.
            if global_step_dir.is_symlink() or global_step_dir.exists():
                if global_step_dir.is_symlink():
                    global_step_dir.unlink()
                else:
                    shutil.rmtree(global_step_dir)
            global_step_dir.symlink_to(hf_ckpt.resolve(), target_is_directory=True)

        # Write marker file
        (output_dir / "latest_checkpointed_iteration.txt").write_text(str(step))


# --- Maze evaluation callback ---


class MazeEvalCallback(TrainerCallback):
    """Run the model on validation mazes and log reward metrics.

    Note: HF Trainer passes the tokenizer as `processing_class=`, not `tokenizer=`.
    We capture references via on_train_begin to avoid kwarg name issues.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.eval_mazes = []
        self._model = None
        self._tokenizer = None
        self._generate_eval_mazes()

    def _generate_eval_mazes(self):
        """Pre-generate fixed eval mazes with deterministic seeds."""
        tile_chars = self.config.tile_chars
        tc = TileConfig(
            PATH=tile_chars["path"],
            LAVA=tile_chars["lava"],
            GOAL=tile_chars["goal"],
            PLAYER=tile_chars["player"],
            mode="emoji" if not tile_chars["path"].isascii() else "letters",
        )
        gen = MazeGenerator(
            size=self.config.maze_size,
            goal_lava_ratio=self.config.goal_lava_ratio,
            tile_config=tc,
        )

        safe_tiles = (tc.PATH, tc.GOAL)
        eval_seed_offset = 2_000_000  # disjoint from train/val
        idx = 0
        while len(self.eval_mazes) < self.config.eval_mazes:
            random.seed(self.config.base_seed + eval_seed_offset + idx)
            maze = gen.generate()
            idx += 1
            # Skip degenerate mazes
            has_safe = any(
                maze.is_in_bounds(maze.start[0] + dx, maze.start[1] + dy)
                and maze.get_tile(maze.start[0] + dx, maze.start[1] + dy) in safe_tiles
                for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]
            )
            if has_safe:
                self.eval_mazes.append(maze)

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        """Capture model and tokenizer references at training start."""
        self._model = model
        self._tokenizer = processing_class

    def on_evaluate(self, args, state, control, **kwargs):
        """Run multi-turn maze games and compute reward."""
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            print("[Maze Eval] SKIPPED: model or tokenizer not available", flush=True)
            return

        model.eval()

        # Temporarily disable gradient checkpointing for generation (KV cache compat)
        gc_enabled = getattr(model, "is_gradient_checkpointing", False)
        if gc_enabled:
            model.gradient_checkpointing_disable()

        rewards = []
        goal_counts = []
        lava_counts = []
        rewards_cfg = {
            "step_penalty": self.config.step_penalty,
            "goal_reward": self.config.goal_reward,
            "lava_penalty": self.config.lava_penalty,
        }

        try:
            with torch.no_grad():
                for maze in self.eval_mazes:
                    direction_order = ["north", "east", "south", "west"]
                    if self.config.shuffle_directions:
                        random.shuffle(direction_order)

                    game = Game(
                        maze,
                        wind_frequency=0.0,
                        melting_path=False,
                        direction_order=direction_order,
                        relative_directions=self.config.relative_directions,
                    )

                    messages = [{"role": "user", "content": game.get_prompt()}]

                    for turn in range(self.config.eval_max_turns):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = tokenizer(text, return_tensors="pt").to(model.device)
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=4,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                        )
                        generated = tokenizer.decode(
                            output_ids[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()

                        direction = parse_direction(generated)
                        if direction is None:
                            break

                        messages.append({"role": "assistant", "content": direction})

                        # Execute move
                        dir_lower = direction.lower()
                        short_map = {"n": "north", "s": "south", "e": "east", "w": "west"}
                        full_dir = short_map.get(dir_lower, dir_lower)
                        if full_dir not in DIRECTIONS:
                            break

                        dx, dy = DIRECTIONS[full_dir]
                        nx, ny = game.position[0] + dx, game.position[1] + dy
                        if not maze.is_in_bounds(nx, ny):
                            break

                        game.move(full_dir)
                        messages.append({"role": "user", "content": game.get_prompt()})

                    trajectory_len = len([m for m in messages if m["role"] == "assistant"])
                    reward = compute_reward(
                        trajectory_len,
                        game.get_goal_visit_count(),
                        game.get_lava_visit_count(),
                        rewards_cfg,
                    )
                    rewards.append(reward)
                    goal_counts.append(game.get_goal_visit_count())
                    lava_counts.append(game.get_lava_visit_count())
        except Exception as e:
            import traceback
            print(f"[Maze Eval] ERROR: {e}", flush=True)
            traceback.print_exc()
            return
        finally:
            # Re-enable gradient checkpointing if it was on
            if gc_enabled:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model.train()

        metrics = {
            "eval/maze_reward_mean": float(np.mean(rewards)),
            "eval/maze_reward_std": float(np.std(rewards)),
            "eval/maze_goals_mean": float(np.mean(goal_counts)),
            "eval/maze_lava_mean": float(np.mean(lava_counts)),
        }

        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass

        print(f"[Maze Eval] step={state.global_step} "
              f"reward={metrics['eval/maze_reward_mean']:.2f} "
              f"goals={metrics['eval/maze_goals_mean']:.2f} "
              f"lava={metrics['eval/maze_lava_mean']:.2f}",
              flush=True)


# --- Main ---


def main():
    config = parse_args()

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / "config.json")

    # Generate or load datasets
    train_dataset = load_or_generate_dataset(config, "train", config.num_train)
    val_dataset = load_or_generate_dataset(config, "val", config.num_val)

    print(f"\nTrain: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # Load tokenizer to configure SFT
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if config.chat_template_from:
        src_tok = AutoTokenizer.from_pretrained(config.chat_template_from)
        assert src_tok.chat_template is not None, (
            f"chat_template_from={config.chat_template_from} has no chat_template"
        )
        tokenizer.chat_template = src_tok.chat_template
        # Align eos_token so it matches the end-of-turn marker in the template
        # (Base has eos=<|endoftext|>, Instruct has eos=<|im_end|>).
        if src_tok.eos_token_id is not None:
            tokenizer.eos_token = src_tok.eos_token
        print(f"Overriding chat_template from {config.chat_template_from}; "
              f"eos_token={tokenizer.eos_token!r} pad_token={tokenizer.pad_token!r}")

    # LoRA config (None if FFT)
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        )
    else:
        lora_config = None

    # Training args
    training_args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_seq_length,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=output_dir.name,
        project=config.wandb_project,
        dataloader_num_workers=0,  # avoid multiprocessing issues on login nodes
    )

    # Callbacks
    callbacks = [LoRACheckpointCallback(config)]
    if config.eval_mazes > 0:
        callbacks.append(MazeEvalCallback(config))

    # Create trainer
    trainer = SFTTrainer(
        model=config.model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Initialize wandb with project info
    if training_args.report_to == "wandb":
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(
                    {"sft_config": json.loads(json.dumps(config.__dict__, default=str))},
                    allow_val_change=True,
                )
        except ImportError:
            pass

    print(f"\nStarting SFT training...")
    print(f"  Model: {config.model_path}")
    if config.use_lora:
        print(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    else:
        print(f"  FFT: full fine-tune (no LoRA)")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps}")
    print()

    trainer.train()

    # Save final checkpoint
    if config.use_lora:
        final_dir = output_dir / "final" / "lora_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(final_dir))
        print(f"\nFinal LoRA adapter saved to {final_dir}")
    else:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"\nFinal full model saved to {final_dir}")


if __name__ == "__main__":
    main()
