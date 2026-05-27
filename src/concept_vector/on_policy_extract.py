"""On-policy concept vector extraction.

Two methods, both consuming the same rollout trajectories so we only roll out once:

1. on_policy: Single 'reward' vector from top10% / bottom10% by reward, captured
   at the LAST assistant turn of each trajectory.

2. on_policy_short_circuit: lava / goal / path concept vectors. Filters rollouts
   where the agent walked PATH-only until the first non-PATH step is LAVA or GOAL.
   Pools:
     - LAVA pool: activations at the move-decision turn that landed on LAVA
     - GOAL pool: activations at the move-decision turn that landed on GOAL
     - PATH pool: activations at the PATH-prefix turns of qualifying lava/goal trajectories
   Concept vectors follow the off-policy convention: mean(positive) - mean(negative).

Both methods capture activations at three positions per recorded turn so we can
empirically determine which position is best:
  A: token immediately preceding the model's first generated token (last prompt token)
  B: first token of analysis (thinking) channel content (None if absent)
  C: first token of non-thinking (final) channel content (typically the direction)

Supports chatml, plain, and harmony formats (gpt-oss / Tinker v5).
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LogitsProcessor

from src.concept_vector.activation_extraction import get_per_sample_position_pre_hook
from src.concept_vector.hook_utils import add_hooks
from src.concept_vector.model_utils import (
    detect_format,
    detect_use_chat_template,
    format_generation_prompt_harmony,
    get_model_block_modules,
    load_base_model,
    load_model_with_lora,
    parse_harmony_output,
)
from src.concept_vector.utils import load_tile_config_from_run_config
from src.maze.game import DIRECTIONS, Game, GameResult
from src.maze.inference import compute_reward, load_run_config, resolve_checkpoint_path
from src.maze.maze import MazeGenerator


# ─── Action masking (chatml/plain only) ─────────────────────────────────────


class DirectionLogitsProcessor(LogitsProcessor):
    """Masks logits to only allow N/E/S/W tokens. Used for chatml/plain rollouts."""

    def __init__(self, direction_token_ids: list[int]):
        assert len(direction_token_ids) == 4, (
            f"Expected 4 direction token IDs, got {len(direction_token_ids)}"
        )
        self.direction_token_ids = direction_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.direction_token_ids:
            mask[:, tid] = 0.0
        return scores + mask


# ─── Format-aware prompt builders ────────────────────────────────────────────


_SHORT_MAP = {"N": "north", "S": "south", "E": "east", "W": "west"}


def _resolve_harmony_special_ids(tokenizer) -> dict:
    """Look up harmony special-token IDs at runtime."""
    ids = {}
    for tok in ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid is not None and tid != tokenizer.unk_token_id, (
            f"harmony token {tok!r} resolved to UNK on this tokenizer"
        )
        ids[tok] = tid
    return ids


def _build_prompt_token_ids(messages: list[dict], tokenizer, format: str) -> torch.LongTensor:
    """Build the pre-generation prompt token IDs for a given format.

    All formats produce a sequence whose final token is the canonical
    "immediately before the model's first generated token" position:
        chatml:  ends after `<|im_start|>assistant\\n`
        harmony: ends after `<|start|>assistant`
        plain:   ends after `\\n\\nI move: ` (space)
    """
    if format == "harmony":
        s = format_generation_prompt_harmony(messages)
        return tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if format == "chatml":
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).squeeze(0)
    if format == "plain":
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                if parts:
                    parts.append("\n\n")
                parts.append(msg["content"])
                parts.append("\n\nI move: ")
            elif msg["role"] == "assistant":
                parts.append(msg["content"])
        plain_text = "".join(parts)
        return tokenizer.encode(
            plain_text, add_special_tokens=False, return_tensors="pt",
        ).squeeze(0)
    raise ValueError(f"Unknown format: {format!r}")


# ─── Harmony channel-position detection ──────────────────────────────────────


def _find_harmony_channel_content_starts(
    generation_token_ids: list[int],
    tokenizer,
    *,
    channel_id: int,
    message_id: int,
) -> dict:
    """Locate the first content-token offset of analysis and final channels.

    The model emits sequences shaped like:
        <|channel|> analysis <|message|> ...content... <|end|>
        <|start|> assistant <|channel|> final <|message|> ...content... <|return|>
    We scan for `<|channel|>` and the next `<|message|>`, decode the chunk between
    them to identify which channel, and record the index of `<|message|> + 1` as
    the first content token.

    Returns dict with keys:
        analysis_idx: int | None  (index in generation_token_ids of first analysis-content token)
        final_idx:    int | None  (index in generation_token_ids of first final-content token)
    Indices are 0-based offsets within `generation_token_ids` (NOT the full
    prompt + generation sequence).
    """
    analysis_idx: int | None = None
    final_idx: int | None = None
    n = len(generation_token_ids)
    i = 0
    while i < n:
        if generation_token_ids[i] == channel_id:
            j = i + 1
            while j < n and generation_token_ids[j] != message_id:
                j += 1
            if j >= n:
                break
            name_tokens = generation_token_ids[i + 1 : j]
            name = tokenizer.decode(name_tokens).strip().lower()
            content_idx = j + 1
            if content_idx < n:
                if "analysis" in name and analysis_idx is None:
                    analysis_idx = content_idx
                elif "final" in name and final_idx is None:
                    final_idx = content_idx
            i = j + 1
        else:
            i += 1
    return {"analysis_idx": analysis_idx, "final_idx": final_idx}


# ─── Per-game record helpers ─────────────────────────────────────────────────


def _make_turn_record(
    *,
    prompt_token_ids: list[int],
    generation_token_ids: list[int],
    direction: str,
    format: str,
    tokenizer,
    harmony_ids: dict | None = None,
    analysis_text: str | None = None,
    final_text: str | None = None,
    failed: bool = False,
) -> dict:
    """Build a per-turn record with capture-position offsets.

    Position offsets are indices within the FULL token sequence
    (prompt_token_ids + generation_token_ids), 0-based.
    """
    n_prompt = len(prompt_token_ids)
    n_gen = len(generation_token_ids)
    full_len = n_prompt + n_gen

    # Position A: last prompt token (immediately before first generated token)
    position_A_idx = n_prompt - 1

    # Positions B and C
    position_B_idx: int | None
    position_C_idx: int | None
    if format == "harmony":
        assert harmony_ids is not None
        offsets = _find_harmony_channel_content_starts(
            generation_token_ids,
            tokenizer,
            channel_id=harmony_ids["<|channel|>"],
            message_id=harmony_ids["<|message|>"],
        )
        position_B_idx = (
            n_prompt + offsets["analysis_idx"]
            if offsets["analysis_idx"] is not None
            else None
        )
        position_C_idx = (
            n_prompt + offsets["final_idx"]
            if offsets["final_idx"] is not None
            else None
        )
    else:
        # chatml/plain: no analysis channel; final content = first generated token
        position_B_idx = None
        position_C_idx = n_prompt if n_gen > 0 else None

    assert 0 <= position_A_idx < full_len, (
        f"position_A_idx {position_A_idx} out of range [0, {full_len})"
    )
    if position_B_idx is not None:
        assert 0 <= position_B_idx < full_len
    if position_C_idx is not None:
        assert 0 <= position_C_idx < full_len

    return {
        "prompt_token_ids": prompt_token_ids,
        "generation_token_ids": generation_token_ids,
        "position_A_idx": position_A_idx,
        "position_B_idx": position_B_idx,
        "position_C_idx": position_C_idx,
        "direction": direction,
        "analysis_text": analysis_text,
        "final_text": final_text,
        "failed": failed,
    }


# ─── On-policy rollouts ─────────────────────────────────────────────────────


def run_on_policy_rollouts(
    model,
    tokenizer,
    mazes: list,
    *,
    rollouts_per_maze: int = 8,
    max_turns: int = 15,
    batch_size: int = 64,
    temperature: float = 1.0,
    rewards_cfg: dict,
    format: str = "chatml",
    max_new_tokens_harmony: int = 64,
) -> list[dict]:
    """Roll out the model on `mazes` and return per-game records.

    For chatml/plain: uses `model.generate(max_new_tokens=1)` with a direction-
    only logits mask (forces N/E/S/W as the first generated token). Fast,
    matches existing on_policy behavior.

    For harmony: uses free-form `model.generate(max_new_tokens=max_new_tokens_harmony)`
    with no logits mask, then parses the output for the direction in the final
    channel. This enables capturing Position B (analysis-channel onset) when the
    model chooses to emit one.
    """
    assert format in ("chatml", "plain", "harmony"), f"Unknown format: {format!r}"
    is_harmony = format == "harmony"

    direction_token_ids: list[int] = []
    id_to_letter: dict[int, str] = {}
    if not is_harmony:
        # Direction-only logits mask path
        direction_tokens: dict[str, int] = {}
        for letter in ["N", "E", "S", "W"]:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"Direction '{letter}' encodes to {len(ids)} tokens: {ids}"
            direction_tokens[letter] = ids[0]
        direction_token_ids = [direction_tokens[d] for d in ["N", "E", "S", "W"]]
        id_to_letter = {v: k for k, v in direction_tokens.items()}
        processor = DirectionLogitsProcessor(direction_token_ids)
    else:
        processor = None

    harmony_ids = _resolve_harmony_special_ids(tokenizer) if is_harmony else None

    # Create games
    games: list[Game] = []
    maze_indices: list[int] = []
    for maze_idx, maze in enumerate(mazes):
        for _ in range(rollouts_per_maze):
            game = Game(maze=maze)
            games.append(game)
            maze_indices.append(maze_idx)

    n = len(games)
    print(f"Created {n} games ({len(mazes)} mazes x {rollouts_per_maze} rollouts), format={format}")

    conversations: list[list[dict]] = [[] for _ in range(n)]
    trajectories: list[list[tuple[int, int]]] = [[g.position] for g in games]
    tile_types: list[list[str]] = [[] for _ in range(n)]
    turn_records: list[list[dict]] = [[] for _ in range(n)]
    active = list(range(n))
    num_turns = [0] * n

    for i in range(n):
        prompt = games[i].get_prompt()
        conversations[i].append({"role": "user", "content": prompt})

    pad_token_id = tokenizer.pad_token_id

    for turn in range(max_turns):
        if not active:
            break
        print(f"  Turn {turn + 1}/{max_turns}: {len(active)} active games")

        newly_finished: list[int] = []
        for batch_start in range(0, len(active), batch_size):
            batch_indices = active[batch_start : batch_start + batch_size]

            # Build prompt token IDs per game (format-specific)
            batch_prompt_ids = [
                _build_prompt_token_ids(conversations[idx], tokenizer, format)
                for idx in batch_indices
            ]

            # Left-pad to uniform length
            max_len = max(len(ids) for ids in batch_prompt_ids)
            padded_ids = []
            attention_masks = []
            pad_lens = []
            for ids in batch_prompt_ids:
                pad_len = max_len - len(ids)
                pad_lens.append(pad_len)
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
                    padded = torch.cat([padding, ids])
                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(len(ids), dtype=torch.long),
                    ])
                else:
                    padded = ids
                    mask = torch.ones(len(ids), dtype=torch.long)
                padded_ids.append(padded)
                attention_masks.append(mask)

            batched_input_ids = torch.stack(padded_ids).to(model.device)
            batched_attention_mask = torch.stack(attention_masks).to(model.device)

            with torch.no_grad():
                if is_harmony:
                    output_ids = model.generate(
                        batched_input_ids,
                        attention_mask=batched_attention_mask,
                        max_new_tokens=max_new_tokens_harmony,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=pad_token_id,
                    )
                else:
                    output_ids = model.generate(
                        batched_input_ids,
                        attention_mask=batched_attention_mask,
                        max_new_tokens=1,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=pad_token_id,
                        logits_processor=[processor],
                    )

            # Process each response
            for j, idx in enumerate(batch_indices):
                # Strip left-padding from prompt slice; everything after max_len is generation.
                prompt_ids = batch_prompt_ids[j].tolist()
                generation_ids = output_ids[j, max_len:].tolist()

                # Determine direction
                if is_harmony:
                    raw = tokenizer.decode(generation_ids, skip_special_tokens=False)
                    channels = parse_harmony_output(raw)
                    final_text = channels.get("final", "").strip()
                    analysis_text = channels.get("analysis", "").strip() or None
                    letter = final_text[0].upper() if final_text else ""
                    failed = letter not in ("N", "E", "S", "W")
                else:
                    # chatml/plain: single token, must be a direction
                    token_id = generation_ids[0] if generation_ids else None
                    if token_id is None or token_id not in id_to_letter:
                        letter = ""
                        failed = True
                    else:
                        letter = id_to_letter[token_id]
                        failed = False
                    final_text = letter
                    analysis_text = None

                if failed:
                    # Record failed turn and finish this game
                    record = _make_turn_record(
                        prompt_token_ids=prompt_ids,
                        generation_token_ids=generation_ids,
                        direction=letter or "",
                        format=format,
                        tokenizer=tokenizer,
                        harmony_ids=harmony_ids,
                        analysis_text=analysis_text,
                        final_text=final_text,
                        failed=True,
                    )
                    turn_records[idx].append(record)
                    tile_types[idx].append("failed")
                    conversations[idx].append({"role": "assistant", "content": letter})
                    newly_finished.append(idx)
                    continue

                # Apply move
                dir_name = _SHORT_MAP[letter]
                dx, dy = DIRECTIONS[dir_name]
                nx, ny = games[idx].position[0] + dx, games[idx].position[1] + dy

                if not games[idx].maze.is_in_bounds(nx, ny):
                    print(f"WARNING: Out-of-bounds move at game {idx}, turn {turn}. Finishing.")
                    record = _make_turn_record(
                        prompt_token_ids=prompt_ids,
                        generation_token_ids=generation_ids,
                        direction=letter,
                        format=format,
                        tokenizer=tokenizer,
                        harmony_ids=harmony_ids,
                        analysis_text=analysis_text,
                        final_text=final_text,
                        failed=True,
                    )
                    turn_records[idx].append(record)
                    tile_types[idx].append("oob")
                    conversations[idx].append({"role": "assistant", "content": letter})
                    newly_finished.append(idx)
                    continue

                result = games[idx].move(dir_name)
                tile_label = (
                    "goal" if result == GameResult.WIN
                    else "lava" if result == GameResult.LOSE
                    else "path"
                )

                num_turns[idx] += 1
                trajectories[idx].append(games[idx].position)
                tile_types[idx].append(tile_label)
                conversations[idx].append({"role": "assistant", "content": letter})

                record = _make_turn_record(
                    prompt_token_ids=prompt_ids,
                    generation_token_ids=generation_ids,
                    direction=letter,
                    format=format,
                    tokenizer=tokenizer,
                    harmony_ids=harmony_ids,
                    analysis_text=analysis_text,
                    final_text=final_text,
                    failed=False,
                )
                turn_records[idx].append(record)

                if turn < max_turns - 1:
                    next_prompt = games[idx].get_prompt()
                    conversations[idx].append({"role": "user", "content": next_prompt})

        active = [i for i in active if i not in newly_finished]

    # Build results
    results = []
    for i in range(n):
        traj_len = len(trajectories[i])
        goal_count = games[i].get_goal_visit_count()
        lava_count = games[i].get_lava_visit_count()
        reward = compute_reward(traj_len, goal_count, lava_count, rewards_cfg)
        results.append({
            "maze_idx": maze_indices[i],
            "messages": conversations[i],
            "reward": reward,
            "trajectory_length": traj_len,
            "goal_visit_count": goal_count,
            "lava_visit_count": lava_count,
            "player_trajectory": trajectories[i],
            "num_turns": num_turns[i],
            "tile_types": tile_types[i],
            "turn_records": turn_records[i],
        })

    return results


# ─── Selection helpers ──────────────────────────────────────────────────────


def select_top_bottom(
    rollout_results: list[dict],
    *,
    top_pct: int,
    bottom_pct: int,
) -> dict:
    """Return last-turn (rollout_idx, turn_idx) for the top_pct% and bottom_pct%
    by reward. Excludes rollouts with zero successful turns."""
    valid = [
        i for i, r in enumerate(rollout_results)
        if r["num_turns"] > 0 and any(not tr["failed"] for tr in r["turn_records"])
    ]
    rewards = [(i, rollout_results[i]["reward"]) for i in valid]
    rewards.sort(key=lambda x: x[1])

    n = len(rewards)
    bottom_k = max(1, int(n * bottom_pct / 100))
    top_k = max(1, int(n * top_pct / 100))

    bottom_indices = [x[0] for x in rewards[:bottom_k]]
    top_indices = [x[0] for x in rewards[-top_k:]]

    def last_record(rollout_idx: int) -> tuple[int, int]:
        # Last successful turn
        records = rollout_results[rollout_idx]["turn_records"]
        for t in range(len(records) - 1, -1, -1):
            if not records[t]["failed"]:
                return (rollout_idx, t)
        # Should not happen given the `valid` filter
        raise AssertionError(f"rollout {rollout_idx} has no successful turn")

    return {
        "top_records": [last_record(i) for i in top_indices],
        "bottom_records": [last_record(i) for i in bottom_indices],
        "top_rewards": [rollout_results[i]["reward"] for i in top_indices],
        "bottom_rewards": [rollout_results[i]["reward"] for i in bottom_indices],
        "n_total_valid": n,
    }


def select_short_circuit(rollout_results: list[dict]) -> dict:
    """Find rollouts whose tile sequence is path*  →  (lava | goal).

    Returns three pools of (rollout_idx, turn_idx) records:
      - lava_records: the LAVA-step turn for trajectories whose first non-PATH step is LAVA
      - goal_records: the GOAL-step turn for trajectories whose first non-PATH step is GOAL
      - path_records: every PATH-prefix turn from BOTH qualifying trajectory sets

    A rollout qualifies if and only if its first non-path tile_type is "lava" or "goal".
    Trajectories whose first non-path step is "failed" or "oob" do NOT qualify, and
    trajectories that never leave PATH within max_turns also do not qualify (no
    LAVA/GOAL anchor for the contrast).
    """
    lava_records: list[tuple[int, int]] = []
    goal_records: list[tuple[int, int]] = []
    path_records: list[tuple[int, int]] = []

    for ridx, r in enumerate(rollout_results):
        tts = r["tile_types"]
        first_nonpath_t = None
        first_nonpath_label = None
        for t, tt in enumerate(tts):
            if tt != "path":
                first_nonpath_t = t
                first_nonpath_label = tt
                break
        if first_nonpath_t is None:
            continue
        if first_nonpath_label not in ("lava", "goal"):
            continue
        # Qualifies. Anchor turn:
        if first_nonpath_label == "lava":
            lava_records.append((ridx, first_nonpath_t))
        else:
            goal_records.append((ridx, first_nonpath_t))
        # Path-prefix turns
        for t in range(first_nonpath_t):
            path_records.append((ridx, t))

    return {
        "lava_records": lava_records,
        "goal_records": goal_records,
        "path_records": path_records,
    }


# ─── Multi-position activation extraction ────────────────────────────────────


_POSITION_KEY = {"A": "position_A_idx", "B": "position_B_idx", "C": "position_C_idx"}


def extract_activations_at_positions(
    model,
    tokenizer,
    rollout_results: list[dict],
    records: list[tuple[int, int]],
    position: str,
    *,
    batch_size: int = 32,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Forward-pass over (rollout_idx, turn_idx) pairs and capture activations
    at the specified per-sample position.

    Args:
        rollout_results: full rollout output list
        records: list of (rollout_idx, turn_idx) pairs to process
        position: one of "A", "B", "C"
        batch_size: forward-pass batch size
        label: human label for tqdm

    Returns:
        (mean: (n_layers, d_model) float64,
         individual: (n_kept, n_layers, d_model) float32 (CPU),
         n_kept: int — records actually used (skipping those with position=None))
    """
    assert position in _POSITION_KEY
    pos_key = _POSITION_KEY[position]

    # Filter records that have this position
    kept: list[tuple[int, int, int, list[int]]] = []
    for ridx, tidx in records:
        rec = rollout_results[ridx]["turn_records"][tidx]
        pos_idx = rec[pos_key]
        if pos_idx is None:
            continue
        full_ids = rec["prompt_token_ids"] + rec["generation_token_ids"]
        kept.append((ridx, tidx, pos_idx, full_ids))

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_kept = len(kept)

    if n_kept == 0:
        return (
            torch.zeros((n_layers, d_model), dtype=torch.float64, device=model.device),
            None,
            0,
        )

    block_modules = get_model_block_modules(model)
    pad_token_id = tokenizer.pad_token_id

    mean_cache = torch.zeros(
        (n_layers, d_model), dtype=torch.float64, device=model.device,
    )
    individual = torch.zeros(
        (n_kept, n_layers, d_model), dtype=torch.float32, device="cpu",
    )

    with torch.no_grad():
        for i in tqdm(range(0, n_kept, batch_size), desc=f"Capture@{position} ({label})"):
            batch = kept[i : i + batch_size]
            B = len(batch)

            seqs = [torch.tensor(b[3], dtype=torch.long) for b in batch]
            unpadded_pos = [b[2] for b in batch]
            seq_lens = [len(s) for s in seqs]
            max_len = max(seq_lens)

            padded = []
            masks = []
            padded_pos = []
            for s, p, L in zip(seqs, unpadded_pos, seq_lens):
                pad_len = max_len - L
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_token_id, dtype=s.dtype)
                    padded.append(torch.cat([padding, s]))
                    masks.append(torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(L, dtype=torch.long),
                    ]))
                else:
                    padded.append(s)
                    masks.append(torch.ones(L, dtype=torch.long))
                # Convert unpadded position to padded position (left-pad shift)
                padded_pos.append(p + pad_len)

            input_ids = torch.stack(padded).to(model.device)
            attn = torch.stack(masks).to(model.device)
            pos_tensor = torch.tensor(padded_pos, dtype=torch.long)

            fwd_pre_hooks = [
                (block_modules[layer], get_per_sample_position_pre_hook(
                    layer=layer,
                    mean_cache=mean_cache,
                    sample_cache=individual,
                    sample_offset=i,
                    n_samples=n_kept,
                    padded_position_indices=pos_tensor,
                ))
                for layer in range(n_layers)
            ]

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(input_ids=input_ids, attention_mask=attn)

    assert not mean_cache.isnan().any(), f"NaN in mean activations at position {position}"
    assert not mean_cache.isinf().any(), f"Inf in mean activations at position {position}"
    return mean_cache, individual, n_kept


# ─── Concept vector assemblers ──────────────────────────────────────────────


def _mean_diff_from_pools(
    mean_pos: torch.Tensor,        # (n_layers, d_model) float64
    counts_pos: int,
    mean_neg_pools: list[tuple[torch.Tensor, int]],  # [(mean, count), ...] — pooled by count-weighted mean
) -> torch.Tensor:
    """Compute mean(positive) - mean(union of negatives), where the union is
    a count-weighted average of pre-computed pool means.
    """
    if counts_pos == 0:
        raise ValueError("Empty positive pool")
    total_neg = sum(c for _, c in mean_neg_pools)
    if total_neg == 0:
        raise ValueError("Empty negative pool")
    neg_mean = sum((m * c for m, c in mean_neg_pools)) / total_neg
    return mean_pos - neg_mean


# ─── Core extraction (rollout once, then dispatch) ───────────────────────────


def run_rollouts_only(
    model,
    tokenizer,
    checkpoint_path,
    *,
    num_mazes: int,
    rollouts_per_maze: int,
    max_turns: int,
    temperature: float,
    rollout_batch_size: int,
    seed: int,
    format: str,
    goal_lava_ratio: float | None = None,
    maze_size: int | None = None,
    max_new_tokens_harmony: int = 64,
) -> dict:
    """Generate mazes + run rollouts. Returns rollout_results + rollout metadata."""
    checkpoint_path = Path(checkpoint_path)
    run_config = load_run_config(checkpoint_path)

    rewards_cfg = run_config["rewards"]
    resolved_maze_size = maze_size or run_config["maze_size"]
    resolved_glr = goal_lava_ratio if goal_lava_ratio is not None else run_config["goal_lava_ratio"]
    tile_config = load_tile_config_from_run_config(checkpoint_path)

    print(f"Generating {num_mazes} mazes (size={resolved_maze_size}, goal_lava_ratio={resolved_glr})")
    generator = MazeGenerator(
        size=resolved_maze_size, goal_lava_ratio=resolved_glr, tile_config=tile_config,
    )
    mazes = []
    for i in range(num_mazes):
        s = seed + i
        random.seed(s)
        mazes.append(generator.generate())

    total_games = num_mazes * rollouts_per_maze
    print(f"\nRunning on-policy rollouts: {total_games} games "
          f"({num_mazes} mazes x {rollouts_per_maze} rollouts), format={format}")

    t_rollout = time.time()
    rollout_results = run_on_policy_rollouts(
        model, tokenizer, mazes,
        rollouts_per_maze=rollouts_per_maze,
        max_turns=max_turns,
        batch_size=rollout_batch_size,
        temperature=temperature,
        rewards_cfg=rewards_cfg,
        format=format,
        max_new_tokens_harmony=max_new_tokens_harmony,
    )
    rollout_time = time.time() - t_rollout

    import statistics as stats
    rewards = [r["reward"] for r in rollout_results]
    goals = [r["goal_visit_count"] for r in rollout_results]
    lava_hits = [r["lava_visit_count"] for r in rollout_results]
    failed_turns = sum(
        sum(1 for tr in r["turn_records"] if tr["failed"]) for r in rollout_results
    )
    print(f"\n{'=' * 60}\nROLLOUT SUMMARY\n{'=' * 60}")
    print(f"  Games:       {len(rollout_results)}")
    print(f"  Reward:      mean={stats.mean(rewards):.2f}  median={stats.median(rewards):.2f}  "
          f"min={min(rewards):.2f}  max={max(rewards):.2f}  std={stats.stdev(rewards) if len(rewards) > 1 else 0:.2f}")
    print(f"  Goals:       mean={stats.mean(goals):.1f}")
    print(f"  Lava hits:   mean={stats.mean(lava_hits):.2f}")
    print(f"  Failed turns: {failed_turns}")
    print(f"  Rollout time: {rollout_time:.1f}s ({rollout_time / max(len(rollout_results), 1) * 1000:.1f} ms/game)")

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "format": format,
        "base_model": run_config["base_model"],
        "num_mazes": num_mazes,
        "rollouts_per_maze": rollouts_per_maze,
        "total_trajectories": len(rollout_results),
        "max_turns": max_turns,
        "temperature": temperature,
        "seed": seed,
        "goal_lava_ratio": resolved_glr,
        "maze_size": resolved_maze_size,
        "max_new_tokens_harmony": max_new_tokens_harmony,
        "tile_config": {
            "mode": tile_config.mode,
            "PATH": tile_config.PATH,
            "LAVA": tile_config.LAVA,
            "GOAL": tile_config.GOAL,
        },
        "rewards_cfg": rewards_cfg,
        "rollout_time_s": round(rollout_time, 1),
        "failed_turns": failed_turns,
    }

    return {"rollout_results": rollout_results, "metadata": metadata}


# ─── Public extraction APIs ──────────────────────────────────────────────────


def extract_on_policy_core(
    model,
    tokenizer,
    checkpoint_path,
    *,
    num_mazes: int = 1024,
    rollouts_per_maze: int = 8,
    max_turns: int = 15,
    temperature: float = 1.0,
    batch_size: int = 128,
    rollout_batch_size: int = 64,
    top_pct: int = 10,
    bottom_pct: int = 10,
    seed: int = 474747,
    store_individual: bool = True,
    use_chat_template: bool | str = True,  # bool for back-compat or "harmony"
    goal_lava_ratio: float | None = None,
    maze_size: int | None = None,
    max_new_tokens_harmony: int = 64,
    rollouts: dict | None = None,
) -> dict:
    """Backwards-compatible regular on_policy extraction (single reward vector
    captured at Position A of the LAST assistant turn).

    For multi-position output use `run_extractions(rollouts, ...)`. This shim
    preserves the original `mean_diff` / `activations_high` / `activations_low`
    / `top_indices` / `bottom_indices` keys that callers (extract_dispatch)
    rely on.
    """
    # Resolve format from use_chat_template flag (legacy)
    if isinstance(use_chat_template, str):
        format = use_chat_template
    elif use_chat_template is True:
        format = "chatml"
    elif use_chat_template is False:
        format = "plain"
    else:
        raise ValueError(f"Invalid use_chat_template: {use_chat_template!r}")

    if rollouts is None:
        rollouts = run_rollouts_only(
            model, tokenizer, checkpoint_path,
            num_mazes=num_mazes,
            rollouts_per_maze=rollouts_per_maze,
            max_turns=max_turns,
            temperature=temperature,
            rollout_batch_size=rollout_batch_size,
            seed=seed,
            format=format,
            goal_lava_ratio=goal_lava_ratio,
            maze_size=maze_size,
            max_new_tokens_harmony=max_new_tokens_harmony,
        )

    rollout_results = rollouts["rollout_results"]
    sel = select_top_bottom(rollout_results, top_pct=top_pct, bottom_pct=bottom_pct)

    # Position A only (back-compat)
    mean_top, ind_top, n_top = extract_activations_at_positions(
        model, tokenizer, rollout_results, sel["top_records"], "A",
        batch_size=batch_size, label="top",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mean_bot, ind_bot, n_bot = extract_activations_at_positions(
        model, tokenizer, rollout_results, sel["bottom_records"], "A",
        batch_size=batch_size, label="bottom",
    )

    # Reshape to (1, n_layers, d_model) for compatibility with existing tooling
    mean_diff = (mean_top - mean_bot).unsqueeze(0)
    if ind_top is not None:
        ind_top = ind_top.unsqueeze(1)  # (n, 1, n_layers, d_model)
    if ind_bot is not None:
        ind_bot = ind_bot.unsqueeze(1)

    metadata = dict(rollouts["metadata"])
    metadata.update({
        "extraction_mode": "on_policy",
        "top_pct": top_pct,
        "bottom_pct": bottom_pct,
        "top_count": n_top,
        "bottom_count": n_bot,
        "top_reward_range": [min(sel["top_rewards"]), max(sel["top_rewards"])],
        "bottom_reward_range": [min(sel["bottom_rewards"]), max(sel["bottom_rewards"])],
        "capture_position": "A",
    })

    return {
        "mean_diff": mean_diff,
        "mean_high": mean_top.unsqueeze(0),
        "mean_low": mean_bot.unsqueeze(0),
        "activations_high": ind_top,
        "activations_low": ind_bot,
        "top_rewards": sel["top_rewards"],
        "bottom_rewards": sel["bottom_rewards"],
        "top_records": sel["top_records"],
        "bottom_records": sel["bottom_records"],
        "rollout_results": rollout_results,
        "metadata": metadata,
        "rollouts": rollouts,
    }


def extract_on_policy_multipos(
    model, tokenizer, checkpoint_path,
    *,
    rollouts: dict,
    top_pct: int = 10,
    bottom_pct: int = 10,
    batch_size: int = 32,
    positions: tuple[str, ...] = ("A", "B", "C"),
) -> dict:
    """Per-position regular on_policy extraction. Returns dict[position] -> {...}.

    Each position entry has the same structure as `extract_on_policy_core`'s
    return (mean_diff, activations_high, activations_low, metadata).
    """
    rollout_results = rollouts["rollout_results"]
    sel = select_top_bottom(rollout_results, top_pct=top_pct, bottom_pct=bottom_pct)

    out: dict[str, dict] = {}
    for pos in positions:
        mean_top, ind_top, n_top = extract_activations_at_positions(
            model, tokenizer, rollout_results, sel["top_records"], pos,
            batch_size=batch_size, label=f"top@{pos}",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mean_bot, ind_bot, n_bot = extract_activations_at_positions(
            model, tokenizer, rollout_results, sel["bottom_records"], pos,
            batch_size=batch_size, label=f"bot@{pos}",
        )

        if n_top == 0 or n_bot == 0:
            print(f"[on_policy] position {pos}: insufficient samples "
                  f"(top={n_top}, bottom={n_bot}); skipping output.")
            out[pos] = None
            continue

        mean_diff = (mean_top - mean_bot).unsqueeze(0)
        ind_top_r = ind_top.unsqueeze(1) if ind_top is not None else None
        ind_bot_r = ind_bot.unsqueeze(1) if ind_bot is not None else None

        meta = dict(rollouts["metadata"])
        meta.update({
            "extraction_mode": "on_policy",
            "capture_position": pos,
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "top_count": n_top,
            "bottom_count": n_bot,
            "top_reward_range": [min(sel["top_rewards"]), max(sel["top_rewards"])],
            "bottom_reward_range": [min(sel["bottom_rewards"]), max(sel["bottom_rewards"])],
        })
        out[pos] = {
            "mean_diff": mean_diff,
            "activations_high": ind_top_r,
            "activations_low": ind_bot_r,
            "metadata": meta,
        }
    return out


def extract_on_policy_short_circuit_multipos(
    model, tokenizer, checkpoint_path,
    *,
    rollouts: dict,
    batch_size: int = 32,
    positions: tuple[str, ...] = ("A", "B", "C"),
) -> dict:
    """Per-position short-circuit extraction.

    Returns dict[position] -> dict[concept] -> {mean_diff, activations_positive,
    activations_negative, metadata} where concept ∈ {lava, goal, path}.
    """
    rollout_results = rollouts["rollout_results"]
    sel = select_short_circuit(rollout_results)

    n_lava = len(sel["lava_records"])
    n_goal = len(sel["goal_records"])
    n_path = len(sel["path_records"])
    print(f"\n[short_circuit] qualifying samples: lava={n_lava}, goal={n_goal}, path={n_path}")

    if n_lava == 0 or n_goal == 0 or n_path == 0:
        print(f"[short_circuit] WARNING: empty pool detected; some concept vectors will be skipped.")

    out: dict[str, dict] = {}
    for pos in positions:
        mean_lava, ind_lava, k_lava = extract_activations_at_positions(
            model, tokenizer, rollout_results, sel["lava_records"], pos,
            batch_size=batch_size, label=f"lava@{pos}",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mean_goal, ind_goal, k_goal = extract_activations_at_positions(
            model, tokenizer, rollout_results, sel["goal_records"], pos,
            batch_size=batch_size, label=f"goal@{pos}",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mean_path, ind_path, k_path = extract_activations_at_positions(
            model, tokenizer, rollout_results, sel["path_records"], pos,
            batch_size=batch_size, label=f"path@{pos}",
        )

        if k_lava == 0 or k_goal == 0 or k_path == 0:
            print(f"[short_circuit] position {pos}: empty pool "
                  f"(lava={k_lava}, goal={k_goal}, path={k_path}); skipping.")
            out[pos] = None
            continue

        # mean(positive) - mean(union of negatives), per off-policy convention
        lava_diff = _mean_diff_from_pools(
            mean_lava, k_lava, [(mean_goal, k_goal), (mean_path, k_path)],
        )
        goal_diff = _mean_diff_from_pools(
            mean_goal, k_goal, [(mean_lava, k_lava), (mean_path, k_path)],
        )
        path_diff = _mean_diff_from_pools(
            mean_path, k_path, [(mean_lava, k_lava), (mean_goal, k_goal)],
        )

        # Reshape activations to (n, 1, n_layers, d_model)
        def _reshape(a):
            return a.unsqueeze(1) if a is not None else None

        # Negative pool concatenation per concept: stack the two non-positive pools
        def _concat_neg(*ts):
            ts = [t for t in ts if t is not None]
            if not ts:
                return None
            return torch.cat(ts, dim=0)

        ind_lava_r = _reshape(ind_lava)
        ind_goal_r = _reshape(ind_goal)
        ind_path_r = _reshape(ind_path)

        base_meta = dict(rollouts["metadata"])
        base_meta.update({
            "extraction_mode": "on_policy_short_circuit",
            "capture_position": pos,
            "n_lava": k_lava,
            "n_goal": k_goal,
            "n_path": k_path,
        })

        concepts = {
            "lava": {
                "mean_diff": lava_diff.unsqueeze(0),
                "activations_positive": ind_lava_r,
                "activations_negative": _concat_neg(ind_goal_r, ind_path_r),
                "metadata": {**base_meta, "concept": "lava",
                              "positive_class": "LAVA",
                              "negative_class": ["GOAL", "PATH"]},
            },
            "goal": {
                "mean_diff": goal_diff.unsqueeze(0),
                "activations_positive": ind_goal_r,
                "activations_negative": _concat_neg(ind_lava_r, ind_path_r),
                "metadata": {**base_meta, "concept": "goal",
                              "positive_class": "GOAL",
                              "negative_class": ["LAVA", "PATH"]},
            },
            "path": {
                "mean_diff": path_diff.unsqueeze(0),
                "activations_positive": ind_path_r,
                "activations_negative": _concat_neg(ind_lava_r, ind_goal_r),
                "metadata": {**base_meta, "concept": "path",
                              "positive_class": "PATH",
                              "negative_class": ["LAVA", "GOAL"]},
            },
        }
        out[pos] = concepts
    return out


# ─── Rollout cache I/O ──────────────────────────────────────────────────────


def save_rollout_cache(rollouts: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Trajectories: turn_records contain raw int lists (not tensors), so JSON works.
    with open(cache_dir / "trajectories.jsonl", "w") as f:
        for r in rollouts["rollout_results"]:
            f.write(json.dumps(r, default=str) + "\n")
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(rollouts["metadata"], f, indent=2)
    print(f"Wrote rollout cache to {cache_dir} ({len(rollouts['rollout_results'])} trajectories)")


def load_rollout_cache(cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    rollout_results = []
    with open(cache_dir / "trajectories.jsonl") as f:
        for line in f:
            rollout_results.append(json.loads(line))
    # JSON re-loads tuples as lists; convert position tuples back to tuples.
    for r in rollout_results:
        r["player_trajectory"] = [tuple(p) for p in r["player_trajectory"]]
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    return {"rollout_results": rollout_results, "metadata": metadata}


# ─── Main (legacy single-method invocation) ──────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="On-policy concept vector extraction (regular only — "
                    "use extract_dispatch.py for short-circuit)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-mazes", type=int, default=1024)
    parser.add_argument("--rollouts-per-maze", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--top-pct", type=int, default=10)
    parser.add_argument("--bottom-pct", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=474747)
    parser.add_argument("--goal-lava-ratio", type=float, default=None)
    parser.add_argument("--maze-size", type=int, default=None)
    parser.add_argument("--base-model-only", action="store_true")
    parser.add_argument("--model-override", type=str, default=None,
                        help="Override the base_model field from run config "
                             "(e.g. for merged-LoRA Tinker checkpoints)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "chatml", "harmony", "plain"])
    parser.add_argument("--max-new-tokens-harmony", type=int, default=64)
    parser.add_argument("--no-store-individual", action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    if args.base_model_only:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    run_config = load_run_config(checkpoint_path)
    base_model = args.model_override or run_config["base_model"]

    if args.format == "auto":
        format = detect_format(str(checkpoint_path))
    else:
        format = args.format

    if args.output:
        output_dir = Path(args.output)
    else:
        parts = []
        p = checkpoint_path
        for _ in range(6):
            if p.name.startswith("global_step_"):
                parts.insert(0, p.name.replace("global_step_", "step"))
            elif p.name.startswith("runs"):
                break
            elif p.name not in ("actor", "lora_adapter", "huggingface"):
                parts.insert(0, p.name)
            p = p.parent
        dirname = "_".join(parts) if parts else "on_policy"
        if args.base_model_only:
            dirname += "_base"
        output_dir = Path("artifacts/concept_vectors") / f"{dirname}_on_policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\n{'=' * 60}")
    if args.base_model_only:
        print(f"Loading base model (no LoRA): {base_model}")
        model, tokenizer = load_base_model(base_model)
    else:
        print(f"Loading model with LoRA from: {checkpoint_path}")
        model, tokenizer = load_model_with_lora(str(checkpoint_path), base_model)
    print(f"{'=' * 60}\n")

    rollouts = run_rollouts_only(
        model, tokenizer, checkpoint_path,
        num_mazes=args.num_mazes,
        rollouts_per_maze=args.rollouts_per_maze,
        max_turns=args.max_turns,
        temperature=args.temperature,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
        format=format,
        goal_lava_ratio=args.goal_lava_ratio,
        maze_size=args.maze_size,
        max_new_tokens_harmony=args.max_new_tokens_harmony,
    )
    save_rollout_cache(rollouts, output_dir / "rollout_cache")

    multipos = extract_on_policy_multipos(
        model, tokenizer, checkpoint_path,
        rollouts=rollouts,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        batch_size=args.batch_size,
    )

    pos_subdirs = {"A": "pre_response", "B": "first_analysis", "C": "first_final"}
    for pos, payload in multipos.items():
        if payload is None:
            continue
        pos_dir = output_dir / pos_subdirs[pos] / "reward"
        pos_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload["mean_diff"].cpu(), pos_dir / "mean_diff.pt")
        if payload["activations_high"] is not None and not args.no_store_individual:
            torch.save(payload["activations_high"].cpu(), pos_dir / "activations_lava.pt")
        if payload["activations_low"] is not None and not args.no_store_individual:
            torch.save(payload["activations_low"].cpu(), pos_dir / "activations_other.pt")
        with open(pos_dir / "metadata.json", "w") as f:
            json.dump(payload["metadata"], f, indent=2)
        print(f"[on_policy] wrote position={pos} → {pos_dir}")

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}\nDONE in {total_time:.1f}s\n{'=' * 60}")


if __name__ == "__main__":
    main()
