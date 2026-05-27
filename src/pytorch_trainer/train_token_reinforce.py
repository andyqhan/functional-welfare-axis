"""Token-level REINFORCE trainer for maze navigation.

Uses per-turn returns-to-go with per-position group baseline instead of
GRPO's trajectory-level advantage. Each action token gets a distinct
advantage reflecting future reward from that point onward.

Usage:
    python -m src.pytorch_trainer.train_token_reinforce [--config-overrides key=value ...]
"""

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from .config import TrainConfig
from .distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_metrics,
    unwrap_model,
    wrap_model_ddp,
)
from .grpo import compute_equalized_entropy, compute_kl_loss
from .metrics import MetricsLogger, Timer, get_gpu_metrics
from .model import load_actor_model, load_reference_model, load_tokenizer
from .rollout import (
    _resolve_direction_tokens,
    compute_log_probs_at_positions,
    generate_mazes,
    prepare_training_batch,
    run_rollout,
)
from .token_reinforce import compute_token_ppo_loss, compute_token_reinforce_advantages
from .train import (
    _get_explicit_cli_overrides,
    init_wandb,
    load_checkpoint,
    parse_args,
    save_checkpoint,
    save_config,
)


def train_step(
    step: int,
    actor_model: torch.nn.Module,
    ref_model: torch.nn.Module | None,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    timer: Timer,
    logger: MetricsLogger | None,
    rank: int = 0,
    world_size: int = 1,
    entropy_coef: float | None = None,
) -> dict:
    """Execute one training step with token-level REINFORCE advantages.

    Same structure as train.py:train_step() but uses per-turn returns-to-go
    with per-position group baseline instead of GRPO trajectory-level advantages.
    """
    if entropy_coef is None:
        entropy_coef = config.entropy_coef
    step_metrics = {}
    local_batch_size = (config.num_prompts // world_size) * config.group_size

    effective_lsf = config.loss_scale_factor // world_size

    raw_model = unwrap_model(actor_model)

    # === Phase 1: Generate mazes ===
    with timer.track("maze_gen"):
        mazes = generate_mazes(config, step, rank=rank, world_size=world_size)
    if logger:
        logger.log_gpu(step, "maze_gen", get_gpu_metrics())

    # === Phase 2: Rollout ===
    with timer.track("rollout"):
        trajectories, rollout_timing = run_rollout(
            raw_model, tokenizer, mazes, config, device,
        )
    if logger:
        logger.log_gpu(step, "rollout", get_gpu_metrics())

    assert len(trajectories) == local_batch_size, (
        f"Expected {local_batch_size} trajectories, got {len(trajectories)}"
    )

    # Log rollout stats
    rewards = [t.reward for t in trajectories]
    turns = [t.num_turns for t in trajectories]
    goals = [t.goal_visits for t in trajectories]
    lavas = [t.lava_visits for t in trajectories]

    term_counts = {"max_turns": 0, "invalid_output": 0, "invalid_move": 0}
    for t in trajectories:
        if t.termination_reason in term_counts:
            term_counts[t.termination_reason] += 1

    invalid_trajs = [t for t in trajectories if t.termination_reason == "invalid_output"]
    if invalid_trajs and is_main_process(rank):
        samples = invalid_trajs[:5]
        print(f"  Step {step}: {len(invalid_trajs)}/{local_batch_size} trajectories terminated by invalid output:")
        for t in samples:
            print(f"    turn {t.num_turns}: output={t.invalid_output_text!r}")

    traj_mean_entropies = [
        np.mean(t.action_entropies) for t in trajectories if t.action_entropies
    ]

    step_metrics.update({
        "rollout/reward_mean": np.mean(rewards),
        "rollout/reward_std": np.std(rewards),
        "rollout/reward_min": np.min(rewards),
        "rollout/reward_max": np.max(rewards),
        "rollout/turns_mean": np.mean(turns),
        "rollout/turns_min": np.min(turns),
        "rollout/turns_max": np.max(turns),
        "rollout/goals_mean": np.mean(goals),
        "rollout/lava_mean": np.mean(lavas),
        "rollout/active_frac": sum(1 for t in trajectories if t.active) / local_batch_size,
        "rollout/term_max_turns": term_counts["max_turns"] / local_batch_size,
        "rollout/term_invalid_output": term_counts["invalid_output"] / local_batch_size,
        "rollout/term_invalid_move": term_counts["invalid_move"] / local_batch_size,
        "rollout/action_entropy_mean": np.mean(traj_mean_entropies) if traj_mean_entropies else 0.0,
        "rollout/action_entropy_std": np.std(traj_mean_entropies) if traj_mean_entropies else 0.0,
    })
    step_metrics.update({f"rollout_timing/{k}": v for k, v in rollout_timing.items()})

    # === Phase 3: Prepare training batch ===
    with timer.track("prepare_batch"):
        batch = prepare_training_batch(trajectories, tokenizer, device)
    if logger:
        logger.log_gpu(step, "prepare_batch", get_gpu_metrics())

    torch.cuda.empty_cache()

    # === Phase 4: Compute reference model log probs (skip if no KL penalty) ===
    if ref_model is not None:
        with timer.track("ref_log_probs"):
            ref_log_probs = compute_log_probs_at_positions(
                ref_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_positions"],
                batch["response_token_ids"],
                batch["response_mask"],
                micro_batch_size=config.ref_micro_batch_size,
            )
        if logger:
            logger.log_gpu(step, "ref_log_probs", get_gpu_metrics())
    else:
        ref_log_probs = None

    # === Phase 5: Compute token-level REINFORCE advantages ===
    with timer.track("advantages"):
        advantages, adv_metrics = compute_token_reinforce_advantages(
            batch["per_turn_rewards"],
            batch["response_mask"],
            group_size=config.group_size,
            norm_by_std=config.norm_adv_by_std,
        )
    step_metrics.update(adv_metrics)

    # Also log per-turn reward distribution
    valid_per_turn = batch["per_turn_rewards"][batch["response_mask"].bool()]
    if valid_per_turn.numel() > 0:
        step_metrics["token_reinforce/per_turn_reward_mean"] = valid_per_turn.mean().item()
        step_metrics["token_reinforce/per_turn_reward_std"] = valid_per_turn.std().item() if valid_per_turn.numel() > 1 else 0.0
        step_metrics["token_reinforce/per_turn_reward_min"] = valid_per_turn.min().item()
        step_metrics["token_reinforce/per_turn_reward_max"] = valid_per_turn.max().item()

    # Resolve direction tokens for equalized entropy (only if enabled)
    direction_token_ids = None
    if config.entropy_coef > 0:
        direction_token_ids = _resolve_direction_tokens(tokenizer, device)

    # === Phase 6: PPO Update (with token-level advantages) ===
    actor_model.train()

    indices = torch.randperm(local_batch_size, device=device)

    total_pg_loss = 0.0
    total_kl_loss = 0.0
    total_eq_entropy = 0.0
    num_mini_batches = 0

    mb_clip_fracs = []
    mb_ratio_means = []
    mb_ppo_kls = []

    for ppo_epoch in range(config.ppo_epochs):
        for mb_start in range(0, local_batch_size, config.mini_batch_size):
            mb_end = min(mb_start + config.mini_batch_size, local_batch_size)
            mb_idx = indices[mb_start:mb_end]
            mb_size = mb_end - mb_start

            mb_input_ids = batch["input_ids"][mb_idx]
            mb_attention_mask = batch["attention_mask"][mb_idx]
            mb_positions = batch["response_positions"][mb_idx]
            mb_token_ids = batch["response_token_ids"][mb_idx]
            mb_mask = batch["response_mask"][mb_idx]
            mb_old_log_probs = batch["old_log_probs"][mb_idx]
            mb_advantages = advantages[mb_idx]  # [mini_batch, max_turns]
            mb_ref_log_probs = ref_log_probs[mb_idx] if ref_log_probs is not None else None
            mb_tile_types = batch["tile_types"][mb_idx] if direction_token_ids is not None else None

            optimizer.zero_grad()
            num_micro = max(1, (mb_size + config.micro_batch_size - 1) // config.micro_batch_size)

            for ub_start in range(0, mb_size, config.micro_batch_size):
                ub_end = min(ub_start + config.micro_batch_size, mb_size)
                is_last_micro = (ub_end >= mb_size)

                with timer.track("actor_forward"):
                    fwd_result = compute_log_probs_at_positions(
                        actor_model,
                        mb_input_ids[ub_start:ub_end],
                        mb_attention_mask[ub_start:ub_end],
                        mb_positions[ub_start:ub_end],
                        mb_token_ids[ub_start:ub_end],
                        mb_mask[ub_start:ub_end],
                        micro_batch_size=config.micro_batch_size,
                        direction_token_ids=direction_token_ids,
                    )
                    if direction_token_ids is not None:
                        ub_log_probs, ub_direction_logits = fwd_result
                    else:
                        ub_log_probs = fwd_result

                with timer.track("loss_compute"):
                    # Token-level PPO loss (advantages are [micro_batch, max_turns])
                    pg_loss, pg_metrics = compute_token_ppo_loss(
                        log_probs=ub_log_probs,
                        old_log_probs=mb_old_log_probs[ub_start:ub_end],
                        advantages=mb_advantages[ub_start:ub_end],
                        response_mask=mb_mask[ub_start:ub_end],
                        clip_ratio=config.clip_ratio,
                        clip_ratio_c=config.clip_ratio_c,
                        loss_scale_factor=effective_lsf,
                    )

                    if mb_ref_log_probs is not None:
                        kl_loss, kl_metrics = compute_kl_loss(
                            log_probs=ub_log_probs,
                            ref_log_probs=mb_ref_log_probs[ub_start:ub_end],
                            response_mask=mb_mask[ub_start:ub_end],
                            kl_type=config.kl_type,
                            loss_scale_factor=effective_lsf,
                        )
                    else:
                        kl_loss = None
                        kl_metrics = {"actor/kl_loss": 0.0, "actor/kl_mean_per_token": 0.0}

                    if direction_token_ids is not None:
                        eq_entropy, eq_metrics = compute_equalized_entropy(
                            ub_direction_logits,
                            mb_tile_types[ub_start:ub_end],
                            mb_mask[ub_start:ub_end],
                            loss_scale_factor=effective_lsf,
                        )
                    else:
                        eq_entropy = None
                        eq_metrics = {"actor/equalized_entropy_bonus": 0.0, "actor/equalized_entropy_per_token": 0.0}

                    total_loss = pg_loss
                    if kl_loss is not None:
                        total_loss = total_loss + config.kl_coef * kl_loss
                    if eq_entropy is not None:
                        total_loss = total_loss - entropy_coef * eq_entropy
                    total_loss = total_loss / num_micro

                no_sync = (
                    nullcontext() if is_last_micro or not hasattr(actor_model, 'no_sync')
                    else actor_model.no_sync()
                )
                with no_sync:
                    with timer.track("backward"):
                        total_loss.backward()

                total_pg_loss += pg_loss.detach().item() / num_micro
                if kl_loss is not None:
                    total_kl_loss += kl_loss.detach().item() / num_micro
                if eq_entropy is not None:
                    total_eq_entropy += eq_entropy.detach().item() / num_micro

            if logger:
                logger.log_gpu(step, "backward", get_gpu_metrics())

            with timer.track("backward"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_model.parameters(), config.max_grad_norm,
                )
                optimizer.step()

            num_mini_batches += 1
            step_metrics.update(pg_metrics)
            step_metrics.update(kl_metrics)
            step_metrics.update(eq_metrics)
            step_metrics["actor/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            mb_clip_fracs.append(pg_metrics["actor/clip_frac"])
            mb_ratio_means.append(pg_metrics["actor/ratio_mean"])
            mb_ppo_kls.append(pg_metrics["actor/ppo_kl"])

    step_metrics["actor/total_pg_loss"] = total_pg_loss / max(num_mini_batches, 1)
    step_metrics["actor/total_kl_loss"] = total_kl_loss / max(num_mini_batches, 1)
    step_metrics["actor/total_eq_entropy"] = total_eq_entropy / max(num_mini_batches, 1)

    step_metrics["actor/clip_frac_first"] = mb_clip_fracs[0]
    step_metrics["actor/clip_frac_last"] = mb_clip_fracs[-1]
    step_metrics["actor/clip_frac_mean"] = sum(mb_clip_fracs) / len(mb_clip_fracs)
    step_metrics["actor/ratio_mean_first"] = mb_ratio_means[0]
    step_metrics["actor/ratio_mean_last"] = mb_ratio_means[-1]
    step_metrics["actor/ratio_mean_mean"] = sum(mb_ratio_means) / len(mb_ratio_means)
    step_metrics["actor/ppo_kl_first"] = mb_ppo_kls[0]
    step_metrics["actor/ppo_kl_last"] = mb_ppo_kls[-1]
    step_metrics["actor/ppo_kl_mean"] = sum(mb_ppo_kls) / len(mb_ppo_kls)

    seq_lens = batch["attention_mask"].sum(dim=1).float()
    step_metrics["data/seq_len_mean"] = seq_lens.mean().item()
    step_metrics["data/seq_len_max"] = seq_lens.max().item()
    step_metrics["data/response_turns_mean"] = batch["response_mask"].sum(dim=1).float().mean().item()

    # Log sample trajectories (rank 0 only)
    if logger:
        sample_indices = random.sample(range(local_batch_size), min(3, local_batch_size))
        sample_records = []
        for idx in sample_indices:
            t = trajectories[idx]
            record = {
                "prompt_idx": t.prompt_idx,
                "num_turns": t.num_turns,
                "reward": t.reward,
                "goal_visits": t.goal_visits,
                "lava_visits": t.lava_visits,
                "player_trajectory": t.player_trajectory,
                "termination_reason": t.termination_reason,
                "per_turn_rewards": t.per_turn_rewards,
            }
            if t.invalid_output_text is not None:
                record["invalid_output_text"] = t.invalid_output_text
            sample_records.append(record)
        logger.log_trajectories(step, sample_records)

    return step_metrics


def main():
    # === Distributed init ===
    rank, world_size, device = init_distributed()

    config = parse_args()

    # Resume: override output_dir and find checkpoint
    start_step = 0
    lora_adapter_path = None
    wandb_run_id = None

    if config.resume_from:
        resume_path = Path(config.resume_from)
        assert resume_path.exists(), f"Resume directory not found: {config.resume_from}"

        orig_config_path = resume_path / "config.json"
        if orig_config_path.exists():
            with open(orig_config_path) as f:
                orig = json.load(f)
            cli_overrides = _get_explicit_cli_overrides()
            for key, value in orig.items():
                if hasattr(config, key) and key not in cli_overrides and key != "resume_from":
                    setattr(config, key, value)
            if is_main_process(rank):
                if cli_overrides - {"resume_from", "output_dir"}:
                    print(f"Resume: CLI overrides applied: {cli_overrides - {'resume_from', 'output_dir'}}")

        config.output_dir = config.resume_from
        start_step, lora_adapter_path, fft_model_path = load_checkpoint(config.resume_from)
        if fft_model_path is not None:
            config.model_path = fft_model_path
            assert not config.use_lora, (
                "Checkpoint contains full-model weights but config.use_lora=True."
            )
        else:
            assert config.use_lora, (
                "Checkpoint contains a LoRA adapter but config.use_lora=False."
            )
        if is_main_process(rank):
            print(f"Resuming from {config.resume_from}, step {start_step}")

        wandb_id_file = resume_path / "wandb_run_id.txt"
        if wandb_id_file.exists():
            wandb_run_id = wandb_id_file.read_text().strip()

    # Setup output directory (rank 0 only)
    is_resume = bool(config.resume_from)
    output_dir = config.output_dir
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        save_config(config, output_dir, allow_overwrite=not is_resume)
    if dist.is_initialized():
        dist.barrier()

    if is_main_process(rank):
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs (DDP)")

    wb_run = None
    if is_main_process(rank):
        wb_run = init_wandb(config, resume_run_id=wandb_run_id)

    logger = MetricsLogger(output_dir) if is_main_process(rank) else None
    timer = Timer()

    # Load models
    with timer.track("load_models"):
        tokenizer = load_tokenizer(config)
        actor_model = load_actor_model(config, device, lora_adapter_path=lora_adapter_path)
        if config.kl_coef > 0:
            ref_model = load_reference_model(config, device)
        else:
            ref_model = None
            print("Skipping reference model (kl_coef=0)")

    if is_main_process(rank):
        print(f"Models loaded in {timer.timings['load_models']:.1f}s")
        actor_model.print_trainable_parameters()

    if config.compile_model:
        if is_main_process(rank):
            print("Compiling models with torch.compile...")
        if ref_model is not None:
            ref_model = torch.compile(ref_model)
        actor_model = torch.compile(actor_model)
        if is_main_process(rank):
            print("Models compiled (will trace on first forward pass)")

    actor_model = wrap_model_ddp(actor_model, device)

    if logger:
        logger.log_gpu(0, "models_loaded", get_gpu_metrics())

    trainable_params = [p for p in actor_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.resume_from and start_step > 0:
        ckpt_dir = Path(config.resume_from) / f"global_step_{start_step}"
        opt_state = torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(opt_state["optimizer"])
        if is_main_process(rank):
            print(f"Loaded optimizer state from step {start_step}")
        del opt_state

    if is_main_process(rank):
        local_batch = (config.num_prompts // world_size) * config.group_size
        print(f"\n{'='*60}")
        print(f"Token-Level REINFORCE Trainer")
        print(f"{'='*60}")
        print(f"Model: {config.model_path}")
        print(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
        if world_size > 1:
            print(f"Batch: {config.num_prompts} prompts x {config.group_size} rollouts = {config.total_batch_size} total")
            print(f"  Per GPU: {config.num_prompts // world_size} prompts x {config.group_size} = {local_batch}")
            print(f"  loss_scale_factor: {config.loss_scale_factor} -> {config.loss_scale_factor // world_size} (adjusted for DDP)")
        else:
            print(f"Batch: {config.num_prompts} prompts x {config.group_size} rollouts = {config.total_batch_size}")
        print(f"Max turns: {config.max_turns}")
        print(f"Steps: {start_step}->{config.num_steps}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

    # Training loop
    for step in range(start_step, config.num_steps):
        timer.reset()
        step_start = time.time()

        if config.cosine_decay:
            progress = step / max(config.num_steps - 1, 1) * config.cosine_decay_factor
            progress = min(progress, 1.0)
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * cosine_mult
            current_entropy_coef = config.entropy_coef * cosine_mult
        else:
            cosine_mult = 1.0
            current_entropy_coef = config.entropy_coef

        metrics = train_step(
            step=step,
            actor_model=actor_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            config=config,
            device=device,
            timer=timer,
            logger=logger,
            rank=rank,
            world_size=world_size,
            entropy_coef=current_entropy_coef,
        )

        step_time = time.time() - step_start
        metrics["timing_s/step"] = step_time
        metrics.update({f"timing_s/{k}": v for k, v in timer.timings.items()})
        metrics["timing_s/per_turn"] = step_time / max(metrics["rollout/turns_mean"], 1)

        metrics["schedule/lr"] = optimizer.param_groups[0]["lr"]
        metrics["schedule/entropy_coef"] = current_entropy_coef
        metrics["schedule/cosine_mult"] = cosine_mult

        metrics["gpu/mem_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        metrics["gpu/mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        metrics["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        metrics = reduce_metrics(metrics, world_size)

        if is_main_process(rank):
            if logger:
                logger.log_step(step, metrics)

            if wb_run is not None:
                wandb.log(metrics, step=step)

            if step % config.log_freq == 0:
                print(
                    f"Step {step:4d} | "
                    f"reward={metrics['rollout/reward_mean']:+7.2f} +- {metrics['rollout/reward_std']:.2f} | "
                    f"turns={metrics['rollout/turns_mean']:.1f} | "
                    f"goals={metrics['rollout/goals_mean']:.2f} | "
                    f"pg_loss={metrics['actor/pg_loss']:.4f} | "
                    f"kl={metrics.get('actor/kl_mean_per_token', 0):.4f} | "
                    f"grad={metrics.get('actor/grad_norm', 0):.3f} | "
                    f"rtg={metrics.get('token_reinforce/return_to_go_mean', 0):+.2f} | "
                    f"time={step_time:.1f}s"
                )

        if config.save_freq > 0 and (step + 1) % config.save_freq == 0:
            if is_main_process(rank):
                save_checkpoint(unwrap_model(actor_model), optimizer, step + 1, output_dir, tokenizer=tokenizer)
            if dist.is_initialized():
                dist.barrier()

        if config.cosine_decay and cosine_mult <= 0:
            if is_main_process(rank):
                print(f"\nLR reached zero at step {step} (cosine_decay_factor={config.cosine_decay_factor}), stopping early.")
            break

    final_step = step + 1 if start_step < config.num_steps else config.num_steps
    if is_main_process(rank):
        save_checkpoint(unwrap_model(actor_model), optimizer, final_step, output_dir, tokenizer=tokenizer)
    if dist.is_initialized():
        dist.barrier()

    if is_main_process(rank) and wb_run is not None:
        wandb.finish()

    cleanup_distributed()

    if is_main_process(rank):
        print(f"\nTraining complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
