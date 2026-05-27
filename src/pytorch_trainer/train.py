"""Main training loop: PyTorch GRPO trainer for maze navigation.

Supports single-GPU and multi-GPU data-parallel training via DDP.

Usage:
    # Single GPU:
    python -m src.pytorch_trainer.train [--config-overrides key=value ...]

    # Multi-GPU (via torchrun):
    torchrun --nproc_per_node=2 -m src.pytorch_trainer.train [...]
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
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

from .config import TILE_PRESETS, TrainConfig
from .distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_metrics,
    unwrap_model,
    wrap_model_ddp,
)
from .grpo import compute_equalized_entropy, compute_grpo_advantages, compute_kl_loss, compute_ppo_loss
from .metrics import MetricsLogger, Timer, get_gpu_metrics
from .model import load_actor_model, load_reference_model, load_tokenizer
from .rollout import (
    _resolve_direction_tokens,
    compute_log_probs_at_positions,
    generate_mazes,
    prepare_training_batch,
    run_rollout,
)
from .thinking_rollout import run_thinking_rollout


def _get_explicit_cli_overrides() -> set[str]:
    """Return the set of config field names explicitly passed on the command line.

    This is used during resume to distinguish user overrides from defaults,
    so we can restore the original config and only apply intentional changes.
    """
    explicit = set()
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if "=" in arg:
            # key=value or --key=value style
            key = arg.split("=", 1)[0].lstrip("-")
            explicit.add(key)
        elif arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            # Check if next arg is a value (not another flag)
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                explicit.add(key)
                i += 1  # skip the value
            else:
                explicit.add(key)  # boolean flag
        i += 1
    return explicit


def parse_args() -> TrainConfig:
    """Parse command-line arguments into TrainConfig."""
    parser = argparse.ArgumentParser(description="PyTorch GRPO Maze Trainer")
    config = TrainConfig()

    # Skip fields with custom CLI handling
    _skip_auto_args = {"action_masking", "melting_path", "shuffle_directions", "relative_directions"}

    # Add all config fields as CLI arguments
    for field_name, field_value in vars(config).items():
        if field_name in _skip_auto_args:
            continue
        if isinstance(field_value, bool):
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=field_value)
        elif isinstance(field_value, dict):
            continue  # skip dict fields (tile_chars)
        else:
            parser.add_argument(f"--{field_name}", type=type(field_value), default=field_value)

    # Ergonomic flags (default-on, use --no-X to disable)
    parser.add_argument("--no-action-masking", action="store_true", default=False)
    parser.add_argument("--no-melting-path", action="store_true", default=False)
    parser.add_argument("--no-shuffle-directions", action="store_true", default=False)
    parser.add_argument("--no-relative-directions", action="store_true", default=False)

    # Tile preset (named mapping) and individual overrides
    parser.add_argument("--tile_preset", type=str, default=None, choices=list(TILE_PRESETS.keys()))
    parser.add_argument("--tile_char_path", type=str, default=None)
    parser.add_argument("--tile_char_lava", type=str, default=None)
    parser.add_argument("--tile_char_goal", type=str, default=None)
    parser.add_argument("--tile_char_player", type=str, default=None)

    # Also support key=value overrides (like Hydra)
    args, unknown = parser.parse_known_args()
    for kv in unknown:
        if "=" in kv:
            key, value = kv.split("=", 1)
            key = key.lstrip("-")
            if hasattr(args, key):
                field_type = type(getattr(args, key))
                setattr(args, key, field_type(value))
            else:
                print(f"Warning: unknown config key '{key}', ignoring")

    # Handle --no-X flags (these features default to True in TrainConfig)
    args.action_masking = not args.no_action_masking
    args.melting_path = not args.no_melting_path
    args.shuffle_directions = not args.no_shuffle_directions
    args.relative_directions = not args.no_relative_directions

    # Build config from parsed args
    _cli_only = {"no_action_masking", "no_melting_path", "no_shuffle_directions", "no_relative_directions", "tile_preset", "tile_char_path", "tile_char_lava", "tile_char_goal", "tile_char_player"}
    config_dict = {k: v for k, v in vars(args).items() if k not in _cli_only and hasattr(config, k)}
    result = TrainConfig(**config_dict)

    # Apply tile preset or individual overrides (mutually exclusive)
    has_char_overrides = any(getattr(args, f"tile_char_{k}", None) is not None for k in ("path", "lava", "goal", "player"))
    if args.tile_preset is not None and has_char_overrides:
        parser.error("--tile_preset and --tile_char_* are mutually exclusive")
    if args.tile_preset is not None:
        result.tile_chars = dict(TILE_PRESETS[args.tile_preset])
        result.tile_mode = "letters" if args.tile_preset in ("abc", "xyz") else "emoji"
    for key in ("path", "lava", "goal", "player"):
        val = getattr(args, f"tile_char_{key}", None)
        if val is not None:
            result.tile_chars[key] = val

    return result


def save_config(config: TrainConfig, output_dir: str, *, allow_overwrite: bool = True):
    """Save config to JSON for reproducibility."""
    path = Path(output_dir) / "config.json"
    if path.exists() and not allow_overwrite:
        return  # don't clobber the original config on resume
    with open(path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)
    print(f"Config saved to {path}")


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
    """Execute one full training step: rollout -> advantage -> loss -> update.

    Args:
        actor_model: may be DDP-wrapped. Unwrapped automatically for rollout.
        logger: MetricsLogger instance, or None for non-main ranks.
        rank: distributed rank (0 for single-GPU).
        world_size: number of GPUs (1 for single-GPU).
        entropy_coef: override for config.entropy_coef (e.g. from cosine schedule).

    Returns:
        Dict of metrics for this step.
    """
    if entropy_coef is None:
        entropy_coef = config.entropy_coef
    step_metrics = {}
    local_batch_size = (config.num_prompts // world_size) * config.group_size

    # Scale loss_scale_factor for DDP gradient correctness:
    # DDP mean-reduces gradients, but local batches have fewer mini-batch
    # iterations (N/world_size). Halving lsf doubles per-step gradient,
    # compensating for halved step count → same total gradient per epoch.
    effective_lsf = config.loss_scale_factor // world_size

    # Per-token normalization: standard GRPO normalizes by response length per sequence.
    # Active when training on thinking tokens (many tokens per trajectory).
    per_token_norm = config.enable_thinking and config.train_on_thinking

    raw_model = unwrap_model(actor_model)

    # === Phase 1: Generate mazes ===
    with timer.track("maze_gen"):
        mazes = generate_mazes(config, step, rank=rank, world_size=world_size)
    if logger:
        logger.log_gpu(step, "maze_gen", get_gpu_metrics())

    # === Phase 2: Rollout ===
    if config.enable_thinking:
        with timer.track("rollout"):
            if config.vllm_merge_lora:
                # Merge LoRA into base weights and save full model for vLLM.
                # Required for mxfp4 models (gpt-oss-20b) where vLLM's
                # enable_lora switches to a buggy Triton MoE backend.

                # Offload training models to CPU to free GPU for vLLM
                raw_model.cpu()
                if ref_model is not None:
                    ref_model.cpu()
                torch.cuda.empty_cache()

                # Merge LoRA into base weights on CPU
                raw_model.base_model.merge_adapter()

                # Save merged model to tmpfs as clean safetensors (no LoRA keys).
                # PEFT's inner model has LoRA-structured keys (base_layer.weight,
                # lora_A, lora_B) that a fresh AutoModelForCausalLM can't load,
                # so we remap keys and save with safetensors directly.
                merged_dir = Path(config.vllm_merge_tmpdir) / f"merged_model_{os.getpid()}"
                merged_dir.mkdir(parents=True, exist_ok=True)

                sd = raw_model.base_model.model.state_dict()
                clean_sd = {}
                for k, v in sd.items():
                    if "lora_A" in k or "lora_B" in k:
                        continue
                    clean_sd[k.replace(".base_layer.", ".")] = v

                # Clone tied weights (e.g. embed_tokens/lm_head) to avoid
                # safetensors error on shared memory
                seen_ptrs = {}
                for k, v in clean_sd.items():
                    ptr = v.data_ptr()
                    if ptr in seen_ptrs:
                        clean_sd[k] = v.clone()
                    seen_ptrs[ptr] = k

                from safetensors.torch import save_file
                save_file(clean_sd, str(merged_dir / "model.safetensors"))
                del clean_sd  # free ~40GB CPU memory

                # Save config (strip quantization_config so vLLM loads as bf16)
                model_config = raw_model.base_model.model.config.to_dict()
                model_config.pop("quantization_config", None)
                with open(merged_dir / "config.json", "w") as f:
                    json.dump(model_config, f, indent=2)
                tokenizer.save_pretrained(str(merged_dir))

                # Unmerge adapter (restore base weights for training)
                raw_model.base_model.unmerge_adapter()

                trajectories, rollout_timing = run_thinking_rollout(
                    model_path=str(merged_dir),
                    lora_adapter_path=None,  # No LoRA — already merged
                    tokenizer=tokenizer,
                    mazes=mazes,
                    config=config,
                    device=device,
                )

                # Cleanup tmpfs
                shutil.rmtree(merged_dir, ignore_errors=True)
            else:
                # Standard path: save LoRA adapter, vLLM loads with enable_lora
                temp_lora_dir = Path(config.output_dir) / "_temp_lora"
                raw_model.save_pretrained(str(temp_lora_dir))

                # Offload training models to CPU to free GPU for vLLM
                raw_model.cpu()
                if ref_model is not None:
                    ref_model.cpu()
                torch.cuda.empty_cache()

                trajectories, rollout_timing = run_thinking_rollout(
                    model_path=config.model_path,
                    lora_adapter_path=str(temp_lora_dir),
                    tokenizer=tokenizer,
                    mazes=mazes,
                    config=config,
                    device=device,
                )

            # Reload models to GPU
            raw_model.to(device)
            if ref_model is not None:
                ref_model.to(device)
    else:
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

    # Termination reason counts
    term_counts = {"max_turns": 0, "invalid_output": 0, "invalid_move": 0}
    for t in trajectories:
        if t.termination_reason in term_counts:
            term_counts[t.termination_reason] += 1

    # Log invalid outputs (print first few per step for debugging)
    invalid_trajs = [t for t in trajectories if t.termination_reason == "invalid_output"]
    if invalid_trajs and is_main_process(rank):
        samples = invalid_trajs[:5]
        print(f"  Step {step}: {len(invalid_trajs)}/{local_batch_size} trajectories terminated by invalid output:")
        for t in samples:
            print(f"    turn {t.num_turns}: output={t.invalid_output_text!r}")

    # Action entropy: per-trajectory mean, then aggregate
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
    # Thinking-specific metrics
    if config.enable_thinking:
        all_thinking_counts = [c for t in trajectories for c in t.thinking_token_counts]
        if all_thinking_counts:
            step_metrics.update({
                "rollout/thinking_tokens_mean": np.mean(all_thinking_counts),
                "rollout/thinking_tokens_max": float(max(all_thinking_counts)),
                "rollout/thinking_truncated_frac": sum(
                    1 for c in all_thinking_counts if c >= config.max_thinking_tokens
                ) / len(all_thinking_counts),
            })
    step_metrics.update({f"rollout_timing/{k}": v for k, v in rollout_timing.items()})

    # === Phase 3: Prepare training batch ===
    with timer.track("prepare_batch"):
        batch = prepare_training_batch(trajectories, tokenizer, device)
    if logger:
        logger.log_gpu(step, "prepare_batch", get_gpu_metrics())

    # Log response token count (useful for monitoring train_on_thinking)
    step_metrics["response_positions/mean"] = batch["response_mask"].sum(dim=1).mean().item()
    step_metrics["response_positions/max"] = batch["response_mask"].sum(dim=1).max().item()

    # Free fragmented memory from rollout's dynamic KV cache
    torch.cuda.empty_cache()

    # === Phase 4a: Compute old_log_probs for thinking rollouts ===
    # Non-thinking rollouts compute these during sampling; thinking rollouts
    # defer to here since vLLM doesn't provide exact full-vocab logprobs.
    if config.enable_thinking and not config.skip_old_log_probs:
        with timer.track("old_log_probs"), torch.no_grad():
            batch["old_log_probs"] = compute_log_probs_at_positions(
                raw_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_positions"],
                batch["response_token_ids"],
                batch["response_mask"],
                micro_batch_size=config.ref_micro_batch_size,
            )
        if logger:
            logger.log_gpu(step, "old_log_probs", get_gpu_metrics())

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

    # === Phase 5: Compute advantages ===
    with timer.track("advantages"):
        advantages, group_metrics = compute_grpo_advantages(
            batch["rewards"],
            group_size=config.group_size,
            norm_by_std=config.norm_adv_by_std,
        )
    step_metrics["advantage/mean"] = advantages.mean().item()
    step_metrics["advantage/std"] = advantages.std().item()
    step_metrics.update(group_metrics)

    # Resolve direction tokens for equalized entropy (only if enabled)
    direction_token_ids = None
    if config.entropy_coef > 0:
        direction_token_ids = _resolve_direction_tokens(tokenizer, device)

    # === Phase 6: PPO Update ===
    actor_model.train()

    # Shuffle batch for mini-batch PPO updates
    indices = torch.randperm(local_batch_size, device=device)

    total_pg_loss = 0.0
    total_kl_loss = 0.0
    total_eq_entropy = 0.0
    num_mini_batches = 0

    # Track per-mini-batch metrics to monitor staleness across updates
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
            mb_advantages = advantages[mb_idx]
            mb_ref_log_probs = ref_log_probs[mb_idx] if ref_log_probs is not None else None
            mb_tile_types = batch["tile_types"][mb_idx] if direction_token_ids is not None else None
            mb_direction_mask = batch["direction_mask"][mb_idx] if direction_token_ids is not None else None

            # Gradient accumulation: forward+backward each micro-batch independently
            # so only one micro-batch's activations are in memory at a time.
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
                    # When skip_old_log_probs is set, use current log_probs as old
                    # (ratio=1.0 always, effectively vanilla clipped policy gradient)
                    ub_old = (
                        ub_log_probs.detach()
                        if config.skip_old_log_probs
                        else mb_old_log_probs[ub_start:ub_end]
                    )
                    pg_loss, pg_metrics = compute_ppo_loss(
                        log_probs=ub_log_probs,
                        old_log_probs=ub_old,
                        advantages=mb_advantages[ub_start:ub_end],
                        response_mask=mb_mask[ub_start:ub_end],
                        clip_ratio=config.clip_ratio,
                        clip_ratio_c=config.clip_ratio_c,
                        loss_scale_factor=effective_lsf,
                        per_token_norm=per_token_norm,
                    )

                    if mb_ref_log_probs is not None:
                        kl_loss, kl_metrics = compute_kl_loss(
                            log_probs=ub_log_probs,
                            ref_log_probs=mb_ref_log_probs[ub_start:ub_end],
                            response_mask=mb_mask[ub_start:ub_end],
                            kl_type=config.kl_type,
                            loss_scale_factor=effective_lsf,
                            per_token_norm=per_token_norm,
                        )
                    else:
                        kl_loss = None
                        kl_metrics = {"actor/kl_loss": 0.0, "actor/kl_mean_per_token": 0.0}

                    # Equalized entropy bonus
                    if direction_token_ids is not None:
                        eq_entropy, eq_metrics = compute_equalized_entropy(
                            ub_direction_logits,
                            mb_tile_types[ub_start:ub_end],
                            mb_mask[ub_start:ub_end],
                            loss_scale_factor=effective_lsf,
                            per_token_norm=per_token_norm,
                            direction_mask=mb_direction_mask[ub_start:ub_end] if mb_direction_mask is not None else None,
                        )
                    else:
                        eq_entropy = None
                        eq_metrics = {"actor/equalized_entropy_bonus": 0.0, "actor/equalized_entropy_per_token": 0.0}

                    # Total loss: pg + kl - entropy (entropy is a bonus, so subtract)
                    total_loss = pg_loss
                    if kl_loss is not None:
                        total_loss = total_loss + config.kl_coef * kl_loss
                    if eq_entropy is not None:
                        total_loss = total_loss - entropy_coef * eq_entropy
                    total_loss = total_loss / num_micro

                # Use DDP no_sync for all but last micro-batch to avoid
                # redundant all-reduce during gradient accumulation
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

            # Gradient clipping and optimizer step after accumulation
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

            # Collect per-mini-batch staleness metrics
            mb_clip_fracs.append(pg_metrics["actor/clip_frac"])
            mb_ratio_means.append(pg_metrics["actor/ratio_mean"])
            mb_ppo_kls.append(pg_metrics["actor/ppo_kl"])

    step_metrics["actor/total_pg_loss"] = total_pg_loss / max(num_mini_batches, 1)
    step_metrics["actor/total_kl_loss"] = total_kl_loss / max(num_mini_batches, 1)
    step_metrics["actor/total_eq_entropy"] = total_eq_entropy / max(num_mini_batches, 1)

    # Per-mini-batch staleness summary (first=freshest, last=stalest)
    step_metrics["actor/clip_frac_first"] = mb_clip_fracs[0]
    step_metrics["actor/clip_frac_last"] = mb_clip_fracs[-1]
    step_metrics["actor/clip_frac_mean"] = sum(mb_clip_fracs) / len(mb_clip_fracs)
    step_metrics["actor/ratio_mean_first"] = mb_ratio_means[0]
    step_metrics["actor/ratio_mean_last"] = mb_ratio_means[-1]
    step_metrics["actor/ratio_mean_mean"] = sum(mb_ratio_means) / len(mb_ratio_means)
    step_metrics["actor/ppo_kl_first"] = mb_ppo_kls[0]
    step_metrics["actor/ppo_kl_last"] = mb_ppo_kls[-1]
    step_metrics["actor/ppo_kl_mean"] = sum(mb_ppo_kls) / len(mb_ppo_kls)

    # Log sequence length stats
    seq_lens = batch["attention_mask"].sum(dim=1).float()
    step_metrics["data/seq_len_mean"] = seq_lens.mean().item()
    step_metrics["data/seq_len_max"] = seq_lens.max().item()
    step_metrics["data/response_turns_mean"] = batch["response_mask"].sum(dim=1).float().mean().item()

    # Log sample trajectories (rank 0 only)
    if logger:
        sample_indices = random.sample(
            range(local_batch_size),
            min(config.log_trajectories_per_step, local_batch_size),
        )
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
            }
            if t.invalid_output_text is not None:
                record["invalid_output_text"] = t.invalid_output_text
            sample_records.append(record)
        logger.log_trajectories(step, sample_records)

    return step_metrics


def save_checkpoint(actor_model, optimizer, step: int, output_dir: str, tokenizer=None):
    """Save checkpoint.

    LoRA path: writes adapter to global_step_N/lora_adapter/.
    FFT path: writes full HF model + tokenizer to global_step_N/ directly.
    """
    from peft import PeftModel

    ckpt_dir = Path(output_dir) / f"global_step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(actor_model, PeftModel):
        lora_dir = ckpt_dir / "lora_adapter"
        actor_model.save_pretrained(str(lora_dir))
    else:
        # FFT: save full model + tokenizer into the step dir so the concept-vector
        # loader (which falls back to reading tokenizer from the checkpoint dir when
        # no adapter_config.json exists) can reconstruct the runtime stack.
        actor_model.save_pretrained(str(ckpt_dir))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(ckpt_dir))

    # Save optimizer state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, ckpt_dir / "optimizer.pt")

    # Update latest checkpoint marker
    with open(Path(output_dir) / "latest_checkpointed_iteration.txt", "w") as f:
        f.write(str(step))

    print(f"  Checkpoint saved at step {step} -> {ckpt_dir}")


def load_checkpoint(resume_dir: str) -> tuple[int, str | None, str | None]:
    """Read checkpoint info from a run directory.

    Returns:
        (step, lora_adapter_path, fft_model_path) — exactly one of the last two
        is non-None, matching the format that was saved.
    """
    resume_path = Path(resume_dir)
    marker = resume_path / "latest_checkpointed_iteration.txt"
    assert marker.exists(), f"No checkpoint found in {resume_dir} (missing latest_checkpointed_iteration.txt)"

    step = int(marker.read_text().strip())
    ckpt_dir = resume_path / f"global_step_{step}"
    assert ckpt_dir.exists(), f"Checkpoint dir not found: {ckpt_dir}"
    assert (ckpt_dir / "optimizer.pt").exists(), f"Optimizer state not found at {ckpt_dir / 'optimizer.pt'}"

    lora_dir = ckpt_dir / "lora_adapter"
    if lora_dir.exists():
        return step, str(lora_dir), None

    # FFT path: look for model weights directly under the step dir.
    fft_markers = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]
    if any((ckpt_dir / m).exists() for m in fft_markers):
        return step, None, str(ckpt_dir)

    raise AssertionError(
        f"No lora_adapter/ or full model weights found under {ckpt_dir}"
    )


def init_wandb(config: TrainConfig, resume_run_id: str | None = None):
    """Initialize wandb logging."""
    if wandb is None:
        print("wandb not installed, skipping wandb logging")
        return None

    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        print("WANDB_API_KEY not set, skipping wandb logging")
        return None

    run_name = Path(config.output_dir).name

    kwargs = dict(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=vars(config),
    )

    if resume_run_id:
        kwargs["id"] = resume_run_id
        kwargs["resume"] = "must"

    run = wandb.init(**kwargs)
    print(f"Wandb run: {run.url}")

    # Save run ID for resume
    run_id_path = Path(config.output_dir) / "wandb_run_id.txt"
    run_id_path.write_text(run.id)

    return run


def main():
    # === Distributed init ===
    rank, world_size, device = init_distributed()

    config = parse_args()

    # FFT + thinking mode would require reworking the vLLM rollout path (currently
    # assumes a LoRA adapter to merge/save). Not needed for the LoRA-control runs,
    # so hard-assert rather than silently break.
    if not config.use_lora:
        assert not config.enable_thinking, (
            "use_lora=False is not wired up for enable_thinking=True rollouts; "
            "the vLLM merge path saves the LoRA adapter."
        )

    # Resume: override output_dir and find checkpoint
    start_step = 0
    lora_adapter_path = None
    wandb_run_id = None

    if config.resume_from:
        resume_path = Path(config.resume_from)
        assert resume_path.exists(), f"Resume directory not found: {config.resume_from}"

        # Load original config so we don't lose model_path, hyperparams, etc.
        orig_config_path = resume_path / "config.json"
        if orig_config_path.exists():
            with open(orig_config_path) as f:
                orig = json.load(f)
            # Find which CLI args the user explicitly set (not defaults)
            cli_overrides = _get_explicit_cli_overrides()
            # Start from original config, apply only explicit CLI overrides
            for key, value in orig.items():
                if hasattr(config, key) and key not in cli_overrides and key != "resume_from":
                    setattr(config, key, value)
            if is_main_process(rank):
                if cli_overrides - {"resume_from", "output_dir"}:
                    print(f"Resume: CLI overrides applied: {cli_overrides - {'resume_from', 'output_dir'}}")

        config.output_dir = config.resume_from
        start_step, lora_adapter_path, fft_model_path = load_checkpoint(config.resume_from)
        if fft_model_path is not None:
            # FFT resume: point model_path at the saved step dir so load_actor_model
            # picks up the trained weights instead of the base model.
            config.model_path = fft_model_path
            assert not config.use_lora, (
                "Checkpoint contains full-model weights but config.use_lora=True. "
                "Resume cannot toggle LoRA/FFT mode — re-train from scratch to switch."
            )
        else:
            assert config.use_lora, (
                "Checkpoint contains a LoRA adapter but config.use_lora=False. "
                "Resume cannot toggle LoRA/FFT mode — re-train from scratch to switch."
            )
        if is_main_process(rank):
            print(f"Resuming from {config.resume_from}, step {start_step}")

        # Try to resume wandb run
        wandb_id_file = resume_path / "wandb_run_id.txt"
        if wandb_id_file.exists():
            wandb_run_id = wandb_id_file.read_text().strip()

    elif config.init_lora_from:
        # Fresh training, but initialize LoRA weights from a saved adapter.
        # Does NOT set start_step, does NOT load optimizer, does NOT resume wandb.
        init_path = Path(config.init_lora_from)
        assert init_path.exists(), f"init_lora_from not found: {config.init_lora_from}"
        # Accept either a directory containing lora_adapter/ or the lora_adapter dir directly.
        if (init_path / "lora_adapter").is_dir():
            lora_adapter_path = str(init_path / "lora_adapter")
        elif (init_path / "adapter_config.json").exists():
            lora_adapter_path = str(init_path)
        else:
            raise AssertionError(
                f"init_lora_from={config.init_lora_from} must contain lora_adapter/ "
                f"or adapter_config.json"
            )
        if is_main_process(rank):
            print(f"Initializing LoRA from {lora_adapter_path} (fresh training from step 0)")

    # Setup output directory (rank 0 only)
    is_resume = bool(config.resume_from)
    output_dir = config.output_dir
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        save_config(config, output_dir, allow_overwrite=not is_resume)
    if dist.is_initialized():
        dist.barrier()

    # Device info
    if is_main_process(rank):
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs (DDP)")

    # Initialize wandb (rank 0 only)
    wb_run = None
    if is_main_process(rank):
        wb_run = init_wandb(config, resume_run_id=wandb_run_id)

    # Initialize logger (rank 0 only)
    logger = MetricsLogger(output_dir) if is_main_process(rank) else None
    timer = Timer()

    # Load models
    with timer.track("load_models"):
        tokenizer = load_tokenizer(config)
        actor_model = load_actor_model(config, device, lora_adapter_path=lora_adapter_path)
        if config.kl_coef > 0:
            # For RL-from-SFT, anchor KL against the SFT-initialized policy
            # (not raw base). Use the same adapter that seeded the actor.
            ref_lora_path = lora_adapter_path if config.init_lora_from else None
            ref_model = load_reference_model(
                config, device, lora_adapter_path=ref_lora_path,
            )
        else:
            ref_model = None
            print("Skipping reference model (kl_coef=0)")

    if is_main_process(rank):
        print(f"Models loaded in {timer.timings['load_models']:.1f}s")
        if config.use_lora:
            actor_model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in actor_model.parameters())
            trainable = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
            print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    if config.compile_model:
        if is_main_process(rank):
            print("Compiling models with torch.compile...")
        if ref_model is not None:
            ref_model = torch.compile(ref_model)
        actor_model = torch.compile(actor_model)
        if is_main_process(rank):
            print("Models compiled (will trace on first forward pass)")

    # Wrap actor in DDP (no-op for single GPU)
    actor_model = wrap_model_ddp(actor_model, device)

    if logger:
        logger.log_gpu(0, "models_loaded", get_gpu_metrics())

    # Optimizer (only LoRA parameters)
    trainable_params = [p for p in actor_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Load optimizer state on resume
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
        print(f"PyTorch GRPO Trainer")
        print(f"{'='*60}")
        print(f"Model: {config.model_path}")
        if config.use_lora:
            print(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
        else:
            print("FFT: full fine-tune (no LoRA)")
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

        # Cosine decay schedule for lr and entropy_coef
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

        # Schedule metrics
        metrics["schedule/lr"] = optimizer.param_groups[0]["lr"]
        metrics["schedule/entropy_coef"] = current_entropy_coef
        metrics["schedule/cosine_mult"] = cosine_mult

        # Memory stats
        metrics["gpu/mem_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        metrics["gpu/mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        metrics["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        # All-reduce metrics across ranks for accurate global stats
        metrics = reduce_metrics(metrics, world_size)

        # Log (rank 0 only)
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
                    f"time={step_time:.1f}s"
                )

        # Checkpoint (rank 0 saves, all ranks barrier after)
        if config.save_freq > 0 and (step + 1) % config.save_freq == 0:
            if is_main_process(rank):
                save_checkpoint(unwrap_model(actor_model), optimizer, step + 1, output_dir, tokenizer=tokenizer)
            if dist.is_initialized():
                dist.barrier()

        # Early exit when cosine decay has driven LR to zero
        if config.cosine_decay and cosine_mult <= 0:
            if is_main_process(rank):
                print(f"\nLR reached zero at step {step} (cosine_decay_factor={config.cosine_decay_factor}), stopping early.")
            break

    # Final checkpoint (step+1 reflects completed steps; falls back to num_steps if loop wasn't entered)
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
