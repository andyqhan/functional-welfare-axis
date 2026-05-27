"""DrGRPO algorithm: advantage computation, PPO loss, and KL penalty."""

import torch
import torch.nn.functional as F


def compute_action_entropy(
    logits: torch.Tensor,
    action_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute entropy of the policy distribution over action tokens.

    H(p) = -sum(p_i * log(p_i)) for p = softmax(logits[action_tokens])

    Max entropy for 4 actions = ln(4) ≈ 1.386 nats (uniform distribution).

    Args:
        logits: [..., vocab_size] raw logits (no temperature applied).
        action_token_ids: [num_actions] token IDs for the action tokens (e.g. N/E/S/W).

    Returns:
        entropy: [...] entropy in nats at each position.
    """
    action_logits = logits[..., action_token_ids]  # [..., num_actions]
    log_probs = F.log_softmax(action_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [...]
    return entropy


def compute_equalized_entropy(
    direction_logits: torch.Tensor,
    tile_types: torch.Tensor,
    response_mask: torch.Tensor,
    loss_scale_factor: int = 2048,
    per_token_norm: bool = False,
    direction_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute equalized entropy: entropy within groups of same-tile-type directions.

    For each position, groups directions by tile type (PATH/LAVA/GOAL). For groups
    of size >= 2, computes entropy of the policy restricted to those directions.
    This encourages uniform probability among equally-valued directions without
    fighting the model's ability to distinguish different tile types.

    When train_on_thinking=True, direction_mask restricts entropy computation to
    direction token positions only (thinking positions get zero entropy).

    Args:
        direction_logits: [batch, max_response_len, 4] raw logits for N/E/S/W.
        tile_types: [batch, max_response_len, 4] tile type per direction
                    (0=OOB, 1=PATH, 2=LAVA, 3=GOAL).
        response_mask: [batch, max_response_len] 1 for valid tokens, 0 for padding.
        loss_scale_factor: denominator for seq-mean-token-sum-norm (ignored when per_token_norm=True).
        per_token_norm: if True, normalize per-sequence by direction count then batch mean.
        direction_mask: [batch, max_response_len] 1 at direction positions, 0 at thinking.
                        If None, all response positions are treated as directions.

    Returns:
        (entropy_bonus, metrics): positive scalar entropy bonus + metrics dict.
            Caller should subtract: total_loss -= entropy_coef * entropy_bonus.
    """
    batch_size, max_response_len, num_dirs = direction_logits.shape
    assert num_dirs == 4
    assert tile_types.shape == (batch_size, max_response_len, 4)
    assert response_mask.shape == (batch_size, max_response_len)

    # Effective mask: only compute entropy at valid direction positions
    if direction_mask is not None:
        effective_mask = response_mask * direction_mask
    else:
        effective_mask = response_mask

    total_entropy = torch.zeros(batch_size, max_response_len, device=direction_logits.device)

    for type_id in (1, 2, 3):  # PATH, LAVA, GOAL
        type_mask = (tile_types == type_id)  # [batch, max_response_len, 4]
        group_size = type_mask.sum(dim=-1)  # [batch, max_response_len]

        # Only compute entropy for groups with >= 2 directions of this type
        has_group = (group_size >= 2)  # [batch, max_response_len]
        if not has_group.any():
            continue

        # Mask out non-group logits with -inf so softmax normalizes within group
        neg_inf = torch.tensor(float("-inf"), device=direction_logits.device, dtype=direction_logits.dtype)
        masked_logits = torch.where(type_mask, direction_logits, neg_inf)

        # Positions with NO directions of this type get all-(-inf) logits, which
        # makes log_softmax produce NaN (0/0).  Replace those rows with zeros so
        # log_softmax yields valid values; the result is zeroed out below anyway.
        any_in_group = type_mask.any(dim=-1, keepdim=True)  # [batch, max_response_len, 1]
        safe_logits = torch.where(any_in_group, masked_logits, torch.zeros_like(masked_logits))

        log_probs = F.log_softmax(safe_logits, dim=-1)  # [batch, max_response_len, 4]
        probs = log_probs.exp()

        # Clamp log_probs to avoid 0 * -inf = NaN gradient issue
        safe_log_probs = log_probs.clamp(min=-100.0)
        elem_entropy = -probs * safe_log_probs  # [batch, max_response_len, 4]
        # Zero out non-group positions
        group_entropy = (elem_entropy * type_mask.float()).sum(dim=-1)  # [batch, max_response_len]

        # Only add entropy for positions that actually have a group of this type
        total_entropy = total_entropy + group_entropy * has_group.float()

    # Aggregate using effective_mask (direction positions only)
    seq_entropy = (total_entropy * effective_mask).sum(dim=1)  # [batch]
    if per_token_norm:
        # Normalize by number of direction positions per sequence
        dir_counts = effective_mask.sum(dim=1).clamp(min=1)
        entropy_bonus = (seq_entropy / dir_counts).mean()
    else:
        entropy_bonus = seq_entropy.sum() / loss_scale_factor

    # Per-token entropy for monitoring (at direction positions)
    num_valid = effective_mask.sum().clamp(min=1)
    entropy_per_token = (total_entropy * effective_mask).sum() / num_valid

    metrics = {
        "actor/equalized_entropy_bonus": entropy_bonus.detach().item(),
        "actor/equalized_entropy_per_token": entropy_per_token.detach().item(),
    }

    return entropy_bonus, metrics


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    norm_by_std: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO advantages.

    Args:
        rewards: [batch_size] scalar reward per trajectory.
        group_size: number of rollouts per prompt. batch_size must be divisible.
        norm_by_std: if True, normalize by group std (standard GRPO).
                     if False, no normalization (DrGRPO).

    Returns:
        (advantages, group_metrics): per-trajectory advantages + group spread stats.
    """
    batch_size = rewards.shape[0]
    assert batch_size % group_size == 0, (
        f"batch_size {batch_size} not divisible by group_size {group_size}"
    )
    num_groups = batch_size // group_size

    # Reshape into groups: [num_groups, group_size]
    grouped = rewards.view(num_groups, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)  # [num_groups, 1]

    # advantage = reward - group_mean
    advantages = grouped - group_means  # [num_groups, group_size]

    # Group spread metrics: how diverse are rewards within each prompt's rollouts?
    group_stds = grouped.std(dim=1)  # [num_groups]
    group_ranges = grouped.max(dim=1).values - grouped.min(dim=1).values  # [num_groups]
    # Degenerate groups: all rollouts got the same reward → zero learning signal
    degenerate_frac = (group_stds < 1e-6).float().mean().item()

    if norm_by_std:
        # Standard GRPO: normalize by group std (degenerate groups get zero advantage)
        group_stds_expanded = group_stds.unsqueeze(1)  # [num_groups, 1]
        advantages = advantages / (group_stds_expanded + 1e-8)

    group_metrics = {
        "group/reward_std_mean": group_stds.mean().item(),
        "group/reward_std_min": group_stds.min().item(),
        "group/reward_std_max": group_stds.max().item(),
        "group/reward_range_mean": group_ranges.mean().item(),
        "group/reward_range_min": group_ranges.min().item(),
        "group/reward_range_max": group_ranges.max().item(),
        "group/degenerate_frac": degenerate_frac,
    }

    return advantages.view(batch_size), group_metrics


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_c: float = 3.0,
    loss_scale_factor: int = 2048,
    per_token_norm: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO clipped policy loss.

    Args:
        log_probs: [batch_size, max_response_len] current policy log probs.
        old_log_probs: [batch_size, max_response_len] generation-time log probs.
        advantages: [batch_size] per-trajectory scalar advantage.
        response_mask: [batch_size, max_response_len] 1 for valid tokens, 0 for padding.
        clip_ratio: PPO clip epsilon (default 0.2).
        clip_ratio_c: dual-clip lower bound (default 3.0).
        loss_scale_factor: denominator for seq-mean-token-sum-norm (ignored when per_token_norm=True).
        per_token_norm: if True, normalize per-sequence by token count then batch mean (standard GRPO).

    Returns:
        (loss, metrics_dict)
    """
    assert clip_ratio_c > 1.0, f"clip_ratio_c must be > 1.0, got {clip_ratio_c}"
    batch_size, max_response_len = log_probs.shape
    assert old_log_probs.shape == (batch_size, max_response_len)
    assert advantages.shape == (batch_size,)
    assert response_mask.shape == (batch_size, max_response_len)

    # Broadcast advantages to token level: [batch_size, max_response_len]
    adv = advantages.unsqueeze(1) * response_mask

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

    # Loss aggregation
    seq_losses = (pg_losses * response_mask).sum(dim=1)  # [batch_size]
    if per_token_norm:
        # Standard GRPO: mean over tokens per sequence, then mean over batch
        seq_lengths = response_mask.sum(dim=1).clamp(min=1)  # [batch_size]
        loss = (seq_losses / seq_lengths).mean()
    else:
        # DrGRPO seq-mean-token-sum-norm
        loss = seq_losses.sum() / loss_scale_factor

    metrics = {
        "actor/pg_loss": loss.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/clip_frac": clip_frac.detach().item(),
        "actor/ratio_mean": (ratio * response_mask).sum().detach().item() / response_mask.sum().clamp(min=1).item(),
    }

    return loss, metrics


def compute_kl_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_type: str = "low_var_kl",
    loss_scale_factor: int = 2048,
    per_token_norm: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute KL divergence penalty between policy and reference.

    Args:
        log_probs: [batch_size, max_response_len] current policy.
        ref_log_probs: [batch_size, max_response_len] frozen reference.
        response_mask: [batch_size, max_response_len] valid token mask.
        kl_type: KL estimator type. "low_var_kl" (k3) recommended.
        loss_scale_factor: denominator for seq-mean-token-sum-norm (ignored when per_token_norm=True).
        per_token_norm: if True, normalize per-sequence by token count then batch mean.

    Returns:
        (kl_loss, metrics_dict)
    """
    assert log_probs.shape == ref_log_probs.shape == response_mask.shape

    if kl_type == "low_var_kl" or kl_type == "k3":
        # k3 estimator: exp(ref - pi) - (ref - pi) - 1, clamped
        kl = ref_log_probs - log_probs
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1.0).contiguous()
        kld = torch.clamp(kld, min=-10.0, max=10.0)
    elif kl_type == "kl" or kl_type == "k1":
        kld = log_probs - ref_log_probs
    elif kl_type == "mse" or kl_type == "k2":
        kld = 0.5 * (log_probs - ref_log_probs).square()
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    # Loss aggregation
    seq_kl = (kld * response_mask).sum(dim=1)  # [batch_size]
    if per_token_norm:
        seq_lengths = response_mask.sum(dim=1).clamp(min=1)
        kl_loss = (seq_kl / seq_lengths).mean()
    else:
        kl_loss = seq_kl.sum() / loss_scale_factor

    # Mean KL per token for monitoring
    mean_kl = (kld * response_mask).sum() / response_mask.sum().clamp(min=1)

    metrics = {
        "actor/kl_loss": kl_loss.detach().item(),
        "actor/kl_mean_per_token": mean_kl.detach().item(),
    }

    return kl_loss, metrics
