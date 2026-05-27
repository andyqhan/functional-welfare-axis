"""Token-level REINFORCE: per-turn returns-to-go with per-position group baseline."""

import torch


def compute_returns_to_go(
    per_turn_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute returns-to-go (reverse cumsum) for per-turn rewards.

    G_t = sum_{k=t}^{T-1} r_k  (gamma=1, no discounting)

    Args:
        per_turn_rewards: [batch, max_turns] per-turn reward signal.
        response_mask: [batch, max_turns] 1 for valid turns, 0 for padding.

    Returns:
        G: [batch, max_turns] returns-to-go at each position.
    """
    masked = per_turn_rewards * response_mask
    G = torch.flip(torch.cumsum(torch.flip(masked, [1]), dim=1), [1])
    return G * response_mask


def compute_token_reinforce_advantages(
    per_turn_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
    norm_by_std: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-turn advantages using returns-to-go and per-position group baseline.

    For each prompt group, computes a baseline at each turn position as the mean
    return-to-go across the group's trajectories (weighted by mask). Advantages
    are returns-to-go minus this per-position baseline.

    Args:
        per_turn_rewards: [batch, max_turns] per-turn rewards.
        response_mask: [batch, max_turns] valid turn mask.
        group_size: rollouts per prompt. batch must be divisible.
        norm_by_std: normalize advantages by per-position group std.

    Returns:
        (advantages, metrics): [batch, max_turns] advantages + metrics dict.
    """
    batch_size, max_turns = per_turn_rewards.shape
    assert batch_size % group_size == 0, (
        f"batch_size {batch_size} not divisible by group_size {group_size}"
    )
    num_groups = batch_size // group_size

    # Compute returns-to-go
    G = compute_returns_to_go(per_turn_rewards, response_mask)

    # Reshape to groups: [num_groups, group_size, max_turns]
    G_grouped = G.view(num_groups, group_size, max_turns)
    mask_grouped = response_mask.view(num_groups, group_size, max_turns)

    # Per-position baseline: weighted mean across group members
    # Only average over active trajectories at each position
    mask_sum = mask_grouped.sum(dim=1, keepdim=True).clamp(min=1)  # [num_groups, 1, max_turns]
    baseline = (G_grouped * mask_grouped).sum(dim=1, keepdim=True) / mask_sum  # [num_groups, 1, max_turns]

    # Advantages
    advantages = (G_grouped - baseline) * mask_grouped  # [num_groups, group_size, max_turns]

    if norm_by_std:
        # Per-position std across group
        sq_diff = ((G_grouped - baseline) ** 2) * mask_grouped
        variance = sq_diff.sum(dim=1, keepdim=True) / mask_sum.clamp(min=1)
        std = (variance + 1e-8).sqrt()
        advantages = advantages / std
        advantages = advantages * mask_grouped

    advantages = advantages.view(batch_size, max_turns)

    # --- Metrics ---
    valid_G = G[response_mask.bool()]
    valid_adv = advantages[response_mask.bool()]

    # Degenerate positions: where all group members have the same return
    pos_stds_sq = ((G_grouped - baseline) ** 2 * mask_grouped).sum(dim=1) / mask_sum.squeeze(1).clamp(min=1)
    pos_stds = pos_stds_sq.sqrt()  # [num_groups, max_turns]
    has_any = mask_grouped.any(dim=1).float()  # [num_groups, max_turns]
    degenerate_frac = ((pos_stds < 1e-6) * has_any).sum() / has_any.sum().clamp(min=1)

    # Per-trajectory advantage variance (measures within-trajectory credit assignment spread)
    # For each trajectory, compute std of advantages across its valid turns
    adv_per_traj_std = []
    for i in range(batch_size):
        valid = advantages[i][response_mask[i].bool()]
        if valid.numel() > 1:
            adv_per_traj_std.append(valid.std().item())
    mean_within_traj_adv_std = sum(adv_per_traj_std) / max(len(adv_per_traj_std), 1) if adv_per_traj_std else 0.0

    metrics = {
        "token_reinforce/return_to_go_mean": valid_G.mean().item() if valid_G.numel() > 0 else 0.0,
        "token_reinforce/return_to_go_std": valid_G.std().item() if valid_G.numel() > 1 else 0.0,
        "token_reinforce/advantage_mean": valid_adv.mean().item() if valid_adv.numel() > 0 else 0.0,
        "token_reinforce/advantage_std": valid_adv.std().item() if valid_adv.numel() > 1 else 0.0,
        "token_reinforce/advantage_abs_mean": valid_adv.abs().mean().item() if valid_adv.numel() > 0 else 0.0,
        "token_reinforce/baseline_mean": baseline.mean().item(),
        "token_reinforce/degenerate_position_frac": degenerate_frac.item(),
        "token_reinforce/within_traj_advantage_std": mean_within_traj_adv_std,
    }

    return advantages, metrics


def compute_token_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_c: float = 3.0,
    loss_scale_factor: int = 2048,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO clipped policy loss with per-turn advantages.

    Same as compute_ppo_loss in grpo.py but advantages are [batch, max_turns]
    instead of [batch]. No unsqueeze needed -- advantages are already per-turn.

    Args:
        log_probs: [batch_size, max_turns] current policy log probs.
        old_log_probs: [batch_size, max_turns] generation-time log probs.
        advantages: [batch_size, max_turns] per-turn advantages.
        response_mask: [batch_size, max_turns] 1 for valid turns, 0 for padding.
        clip_ratio: PPO clip epsilon (default 0.2).
        clip_ratio_c: dual-clip lower bound (default 3.0).
        loss_scale_factor: denominator for seq-mean-token-sum-norm.

    Returns:
        (loss, metrics_dict)
    """
    assert clip_ratio_c > 1.0, f"clip_ratio_c must be > 1.0, got {clip_ratio_c}"
    batch_size, max_turns = log_probs.shape
    assert old_log_probs.shape == (batch_size, max_turns)
    assert advantages.shape == (batch_size, max_turns)
    assert response_mask.shape == (batch_size, max_turns)

    # Advantages already per-turn, just mask padding
    adv = advantages * response_mask

    # Importance sampling ratio
    negative_approx_kl = log_probs - old_log_probs
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # Approximate KL divergence for monitoring
    ppo_kl = ((-negative_approx_kl) * response_mask).sum() / response_mask.sum().clamp(min=1)

    # Standard PPO clipped loss
    pg_losses1 = -adv * ratio
    pg_losses2 = -adv * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    clip_pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Dual-clip for negative advantages (prevents too-large negative updates)
    pg_losses_dual = -adv * clip_ratio_c
    clip_pg_losses_dual = torch.minimum(pg_losses_dual, clip_pg_losses)

    # Select: dual-clip for negative advantages, standard clip for positive
    pg_losses = torch.where(adv < 0, clip_pg_losses_dual, clip_pg_losses)

    # Clip fraction for monitoring
    clip_frac = ((pg_losses2 > pg_losses1).float() * response_mask).sum() / response_mask.sum().clamp(min=1)

    # seq-mean-token-sum-norm: sum per sequence, then divide by constant
    seq_losses = (pg_losses * response_mask).sum(dim=1)  # [batch_size]
    loss = seq_losses.sum() / loss_scale_factor

    metrics = {
        "actor/pg_loss": loss.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/clip_frac": clip_frac.detach().item(),
        "actor/ratio_mean": (ratio * response_mask).sum().detach().item() / response_mask.sum().clamp(min=1).item(),
    }

    return loss, metrics
