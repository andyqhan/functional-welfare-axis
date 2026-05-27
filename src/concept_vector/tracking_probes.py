"""Tracking-evidence probes for the functional welfare axis.

For a given (model, eval) pair, load the cached α=0 generations from the
existing concept-vector pipelines, re-tokenize each (prompt, response) under
the model's chat template, run a single forward pass per batch, capture the
residual stream at the layers used for steering, and project onto both the
trained reward vectors (V_lava / V_goal extracted from the maze-trained
checkpoint) and the corresponding baselines (V_lava / V_goal extracted from
the maze-naive base model via the same pipeline).

Writes two parquets per (model, eval) under
``<artifact>/<method>/tracking_probes/<eval>/``:

* ``projections.parquet`` — one row per (prompt_idx, rep_idx). Stores
  tokens, token_ids, assistant_mask, and one float-list per vector. Big.
* ``joined.parquet`` — per-response aggregates (mean/max/last over the
  assistant span) joined with the eval-specific judge fields (sentiment
  score, grade, p_true, refusal_class, etc.). Cheap to scan.

Plus ``metadata.json`` with the layers used, vector source paths, source
generation file, tokenizer hash, and runtime stats.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .explore import auto_select_best_layer
from .hook_utils import add_hooks
from .model_utils import get_model_block_modules, load_model_auto

log = logging.getLogger(__name__)


EVAL_NAMES = ("sentiment", "backtracking", "refusal", "simpleqa", "mmlu",
              "maze_trajectory")


# ─── Loaders for cached α=0 rollouts ────────────────────────────────────────


@dataclasses.dataclass
class Rollout:
    """One (prompt, response) generated at α=0, with all judge fields."""

    prompt_idx: int
    rep_idx: int
    prompt: str
    response: str
    # Eval-specific metadata fields go in ``meta`` and are written out to
    # ``joined.parquet`` next to per-token aggregates.
    meta: dict[str, Any]


def _pick_results_file(concept_dir: Path, prefix: str, benchmark: str | None = None) -> Path:
    """Return the most-recently-modified ``<prefix>_analysis_results_*.json`` in
    ``concept_dir``. If ``prefix == 'simpleqa'``, the corresponding
    ``_analysis_config_*.json`` is consulted and only files whose
    ``benchmark`` field matches ``benchmark`` ('simpleqa' or 'mmlu') are kept.
    """
    candidates = sorted(
        concept_dir.glob(f"{prefix}_analysis_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {prefix}_analysis_results_*.json under {concept_dir}"
        )

    if prefix != "simpleqa":
        return candidates[0]

    if benchmark is None:
        raise ValueError("benchmark must be 'simpleqa' or 'mmlu' for SimpleQA prefix")

    for cand in candidates:
        cfg_path = cand.with_name(
            cand.name.replace("_results_", "_config_")
        )
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        cfg_bench = cfg.get("benchmark")
        # Older runs (pre-2026-04) didn't write a benchmark field and were
        # SimpleQA-only by category. Fall back on n=1000 (SimpleQA) vs
        # n=3420 (MMLU) heuristic only as last resort.
        if cfg_bench == benchmark:
            return cand
    # Heuristic fallback
    for cand in candidates:
        try:
            data = json.loads(cand.read_text())
        except json.JSONDecodeError:
            continue
        n = len(data) if isinstance(data, list) else 0
        if benchmark == "mmlu" and n > 2000:
            return cand
        if benchmark == "simpleqa" and 500 <= n <= 1500:
            return cand
    raise FileNotFoundError(
        f"No {prefix}_analysis_results_*.json matching benchmark={benchmark} in {concept_dir}"
    )


def _load_rollouts_perrep(json_path: Path, judge_fields: tuple[str, ...]) -> list[Rollout]:
    """Schema for sentiment/backtracking/refusal: list of dicts with
    ``analysis['0.0']`` = list of {response, ...judge_fields}.
    """
    data = json.loads(json_path.read_text())
    rollouts: list[Rollout] = []
    for prompt_idx, entry in enumerate(data):
        prompt = entry["prompt"]
        category = entry.get("category", "")
        analysis = entry.get("analysis", {})
        # The alpha key is stringified; α=0 may show as '0.0' or '0'.
        reps = analysis.get("0.0") or analysis.get("0") or []
        for rep_idx, rep in enumerate(reps):
            resp = rep.get("response")
            if not resp:
                continue
            meta = {"prompt_idx": prompt_idx, "rep_idx": rep_idx,
                    "category": category, "target": entry.get("target")}
            for f in judge_fields:
                meta[f] = rep.get(f)
            rollouts.append(Rollout(
                prompt_idx=prompt_idx, rep_idx=rep_idx,
                prompt=prompt, response=resp, meta=meta,
            ))
    return rollouts


def _load_rollouts_single(json_path: Path) -> list[Rollout]:
    """Schema for SimpleQA/MMLU: list of dicts with top-level ``response``
    and ``calibration['0.0']``."""
    data = json.loads(json_path.read_text())
    rollouts: list[Rollout] = []
    for prompt_idx, entry in enumerate(data):
        resp = entry.get("response")
        if not resp:
            continue
        cal0 = (entry.get("calibration") or {}).get("0.0") \
               or (entry.get("calibration") or {}).get("0") or {}
        meta = {
            "prompt_idx": prompt_idx,
            "rep_idx": 0,
            "category": entry.get("category", ""),
            "target": entry.get("target"),
            "grade": entry.get("grade"),
            "parse_failed": entry.get("parse_failed"),
            "p_true": cal0.get("p_true"),
            "p_false": cal0.get("p_false"),
            "logprob_true": cal0.get("logprob_true"),
            "logprob_false": cal0.get("logprob_false"),
        }
        rollouts.append(Rollout(
            prompt_idx=prompt_idx, rep_idx=0,
            prompt=entry["prompt"], response=resp, meta=meta,
        ))
    return rollouts


def _load_rollouts_maze(parquet_path: Path) -> list[Rollout]:
    """Schema for office-emoji maze trajectories (multi-turn convo). Each row
    becomes one Rollout whose ``prompt`` field is a JSON-encoded list of
    messages (we sidestep the (prompt:str, response:str) shape used by
    downstream evals — see ``_build_input_ids`` for the maze branch that
    deserializes and chat-templates the full multi-turn convo).
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    rollouts: list[Rollout] = []
    for prompt_idx, row in df.iterrows():
        # Serialize the trajectory as a list of {role, content} dicts; the
        # forward pass branch deserializes via ``json.loads``.
        traj = row["trajectory"]
        msgs = [{"role": m["role"], "content": m["content"]} for m in traj]
        # Marker: empty response means "no separate response, the whole convo
        # IS the input"; assistant_mask is computed from chat-template token
        # spans during _build_input_ids.
        meta = {
            "prompt_idx": int(prompt_idx),
            "rep_idx": 0,
            "final_tile": row["final_tile"],
            "num_steps": int(row["num_steps"]),
        }
        rollouts.append(Rollout(
            prompt_idx=int(prompt_idx), rep_idx=0,
            prompt=json.dumps(msgs), response="", meta=meta,
        ))
    return rollouts


def load_rollouts(eval_name: str, concept_dir: Path) -> tuple[list[Rollout], Path]:
    """Return α=0 rollouts and the source JSON path. ``concept_dir`` is any
    one concept's directory under the artifact (the α=0 generations are the
    same across concepts; we pick whichever is freshest)."""
    if eval_name == "maze_trajectory":
        # `concept_dir` here is actually the parquet path, passed through.
        # The CLI driver translates an --artifact-agnostic --trajectories-path
        # into this argument.
        if concept_dir.is_dir():
            # Default: project-local office-emoji parquet
            cands = [
                Path("datasets/concept_vector_big_office.parquet"),
                Path("datasets/concept_vector_big.parquet"),
            ]
            parquet = next((p for p in cands if p.exists()), None)
            assert parquet is not None, f"no maze trajectory parquet found among {cands}"
        else:
            parquet = concept_dir
        return _load_rollouts_maze(parquet), parquet
    if eval_name == "sentiment":
        path = _pick_results_file(concept_dir, "sentiment")
        return _load_rollouts_perrep(
            path,
            ("sentiment_score", "parse_failed", "exclamation_count", "emoji_count"),
        ), path
    if eval_name == "backtracking":
        path = _pick_results_file(concept_dir, "backtracking")
        return _load_rollouts_perrep(
            path,
            ("backtracking_class", "parse_failed", "is_correct", "extracted_answer"),
        ), path
    if eval_name == "refusal":
        path = _pick_results_file(concept_dir, "refusal")
        return _load_rollouts_perrep(
            path, ("refusal_class", "parse_failed"),
        ), path
    if eval_name == "simpleqa":
        path = _pick_results_file(concept_dir, "simpleqa", benchmark="simpleqa")
        return _load_rollouts_single(path), path
    if eval_name == "mmlu":
        path = _pick_results_file(concept_dir, "simpleqa", benchmark="mmlu")
        return _load_rollouts_single(path), path
    raise ValueError(f"Unknown eval {eval_name!r}")


# ─── Concept vector loading ─────────────────────────────────────────────────


@dataclasses.dataclass
class LoadedVector:
    name: str            # e.g. "v_lava_trained"
    cv_dir: Path
    layer: int           # ℓ* recovered from metrics.json
    tensor_d: torch.Tensor  # shape (d_model,) — slice at (position=0, layer=ℓ*)


def load_vector(cv_dir: Path, name: str) -> LoadedVector:
    cv_path = cv_dir / "mean_diff.pt"
    assert cv_path.exists(), f"missing {cv_path}"
    layer = auto_select_best_layer(cv_dir)
    assert layer is not None, f"no metrics.json under {cv_dir}"
    full = torch.load(cv_path, map_location="cpu", weights_only=True)
    # Shape (n_positions, n_layers, d_model); take position=0 at chosen layer.
    assert full.dim() == 3, f"unexpected vector shape {tuple(full.shape)} for {cv_path}"
    v = full[0, layer, :].float().contiguous()
    return LoadedVector(name=name, cv_dir=cv_dir, layer=layer, tensor_d=v)


# ─── Per-token capture hook ─────────────────────────────────────────────────


def _make_per_token_pre_hook(layer_idx: int, cache: dict[int, torch.Tensor]):
    """Forward pre-hook that captures residual-stream activations entering
    transformer block ``layer_idx``. Stored on the model's device in its
    native dtype (typically bf16) — projection happens on-device afterwards.
    """
    def hook_fn(module, args):
        activation = args[0]
        # (B, L, d_model)
        cache[layer_idx] = activation.detach()
    return hook_fn


def _left_pad_stack(ids_list: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in ids_list)
    padded = []
    attn = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded.append(torch.cat([
                torch.full((pad_len,), pad_id, dtype=ids.dtype), ids,
            ]))
            attn.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ]))
        else:
            padded.append(ids)
            attn.append(torch.ones(len(ids), dtype=torch.long))
    return torch.stack(padded), torch.stack(attn)


# ─── Tokenization (prompt + response, with assistant-mask) ──────────────────


def _build_input_ids(
    tokenizer, prompt: str, response: str,
) -> tuple[torch.Tensor, int]:
    """Return (full_input_ids: 1D Long, prompt_len) so that
    ``input_ids[prompt_len:]`` is the assistant response. ChatML path:
    apply_chat_template to a single user turn with add_generation_prompt=True
    (matches what explore.py uses to seed generation), then concat the
    response token ids on top.

    Maze-trajectory branch: when ``prompt`` is a JSON-encoded list of
    {role, content} messages AND ``response`` is empty, treat the full
    multi-turn convo as the input (no separate response) and synthesize a
    "prompt_len = 0" so the per-token aggregates cover EVERY position.
    """
    if response == "" and prompt.startswith("[") and "role" in prompt[:50]:
        try:
            messages = json.loads(prompt)
        except json.JSONDecodeError:
            messages = None
        if isinstance(messages, list) and messages and isinstance(messages[0], dict) \
           and "role" in messages[0]:
            full_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True,
                return_tensors="pt",
            )[0]
            # Whole sequence is "of interest" — assistant_mask captures every
            # position; downstream aggregates use it as-is.
            return full_ids, 0

    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )[0]
    resp_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt")[0]
    full = torch.cat([prompt_ids, resp_ids])
    return full, int(prompt_ids.shape[0])


# ─── Feedback-probe tokenization ──────────────────────────────────────────────

FEEDBACK_TEXTS = {"right": "That's right", "wrong": "That's wrong"}


def _build_feedback_input_ids(
    tokenizer, prompt: str, response: str, feedback_text: str,
) -> tuple[torch.Tensor, int]:
    """Build a 3-turn conversation ending with a generation prompt::

        user: <question>
        assistant: <answer>
        user: <feedback_text>
        assistant:   ← prompt_len points here; assistant_mask covers only
                       the generation-prompt tokens (typically 3 for ChatML).
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": feedback_text},
    ]
    without_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_tensors="pt",
    )[0]
    with_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )[0]
    prompt_len = int(without_gen.shape[0])
    return with_gen, prompt_len


def _expand_rollouts_for_feedback(
    rollouts: list[Rollout],
) -> list[tuple[Rollout, str, str]]:
    """Expand each rollout into two (rollout, label, text) triples — one per
    feedback variant ("right" / "wrong")."""
    expanded: list[tuple[Rollout, str, str]] = []
    for r in rollouts:
        for label, text in FEEDBACK_TEXTS.items():
            expanded.append((r, label, text))
    return expanded


# ─── Main driver ────────────────────────────────────────────────────────────


def _tokenizer_hash(tokenizer) -> str:
    # Quick fingerprint so consumers can detect schema drift.
    vocab_str = "|".join(sorted(tokenizer.get_vocab().keys())[:200])
    return hashlib.sha1(vocab_str.encode()).hexdigest()[:12]


_SINK_POS_THRESHOLD = 4
# Conservative magnitude cap — filters the obvious register / attention-sink
# tokens (start-of-sequence, special tokens) where projection blows up to
# thousands, while keeping the normal assistant-content range. V_goal_trained
# regularly projects to 30-80 on ordinary tokens, so a stricter threshold
# would throw away signal. Robust per-response is the median (also reported).
_SINK_MAG_THRESHOLD = 200.0


def _agg(proj_full: np.ndarray, mask: np.ndarray,
         typical_mask: np.ndarray | None = None) -> dict[str, float]:
    """Aggregate per-token projections over the assistant span.

    ``typical_mask`` (if given) further restricts the population to positions
    where every captured projection's |proj| is below the register-token
    threshold (~30) — that mask is computed once per response, across all
    vectors, by the caller. Both raw and "typical" stats are returned so
    consumers can pick what matches their use case.
    """
    sub = proj_full[mask]
    if sub.size == 0:
        out = {"mean": float("nan"), "max": float("nan"), "min": float("nan"),
               "median": float("nan"), "last": float("nan"),
               "abs_max": float("nan"), "n_tok": 0,
               "mean_typical": float("nan"), "n_tok_typical": 0}
        return out
    out = {
        "mean": float(sub.mean()),
        "max": float(sub.max()),
        "min": float(sub.min()),
        "median": float(np.median(sub)),
        "last": float(sub[-1]),
        "abs_max": float(np.abs(sub).max()),
        "n_tok": int(sub.size),
    }
    if typical_mask is not None and typical_mask.any():
        sub_t = proj_full[typical_mask]
        out["mean_typical"] = float(sub_t.mean()) if sub_t.size else float("nan")
        out["n_tok_typical"] = int(sub_t.size)
    else:
        out["mean_typical"] = float("nan")
        out["n_tok_typical"] = 0
    return out


def run_tracking_probes(
    *,
    eval_name: str,
    rollouts_source_dir: Path,   # e.g. <artifact>/off_policy/lava
    vectors: list[LoadedVector],  # 2 trained + 2 baseline = 4 typically
    model_path: str,              # checkpoint path or HF id (e.g. "Qwen/Qwen3-4B-Instruct-2507")
    output_dir: Path,
    base_model_path: str | None = None,
    feedback: bool = False,
    batch_size: int = 4,
    max_total_len: int = 4096,
    max_rollouts: int | None = None,
    save_every: int = 200,
    device: str | None = None,
) -> dict[str, Any]:
    """Top-level: load model+tokenizer, load α=0 rollouts, forward pass per
    batch capturing residual stream at the union of vector layers, project
    onto each vector, save projections+joined parquets to ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading rollouts: %s from %s", eval_name, rollouts_source_dir)
    rollouts, source_json = load_rollouts(eval_name, rollouts_source_dir)
    if max_rollouts is not None:
        rollouts = rollouts[:max_rollouts]
    log.info("Got %d rollouts from %s", len(rollouts), source_json.name)

    # Sanity: all vectors must share dtype/shape and reference layers
    # available on the chosen model.
    layers_needed = sorted({v.layer for v in vectors})
    log.info("Vectors:")
    for v in vectors:
        log.info("  %s @ layer %d (cv_dir=%s)", v.name, v.layer, v.cv_dir)
    log.info("Layers to capture: %s", layers_needed)

    # Load model + tokenizer
    log.info("Loading model from %s (base=%s)", model_path, base_model_path)
    t0 = time.time()
    model, tokenizer = load_model_auto(model_path, base_model_path)
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    log.info("Model loaded in %.1fs on %s", time.time() - t0, device)
    log.info("num_hidden_layers=%d, hidden_size=%d",
             model.config.num_hidden_layers, model.config.hidden_size)
    for v in vectors:
        assert v.tensor_d.shape[0] == model.config.hidden_size, (
            f"vector {v.name} has d={v.tensor_d.shape[0]} != model hidden {model.config.hidden_size}"
        )
        assert v.layer < model.config.num_hidden_layers, (
            f"vector {v.name} wants layer {v.layer} but model has {model.config.num_hidden_layers}"
        )

    # Move vectors to device for projection
    block_modules = get_model_block_modules(model)
    v_by_layer: dict[int, list[LoadedVector]] = {}
    for v in vectors:
        v_by_layer.setdefault(v.layer, []).append(v)
    v_on_device: dict[str, torch.Tensor] = {
        v.name: v.tensor_d.to(device=device, dtype=torch.float32) for v in vectors
    }

    pad_id = tokenizer.pad_token_id

    # Pre-tokenize all rollouts so we can sort by length for efficient packing.
    # Each entry is (rollout, full_ids, prompt_len, feedback_label).
    # feedback_label is "" for non-feedback mode.
    log.info("Tokenizing %d rollouts (feedback=%s)...", len(rollouts), feedback)
    tokenized: list[tuple[Rollout, torch.Tensor, int, str]] = []
    too_long = 0
    if feedback:
        expanded = _expand_rollouts_for_feedback(rollouts)
        for r, fb_label, fb_text in expanded:
            ids, plen = _build_feedback_input_ids(
                tokenizer, r.prompt, r.response, fb_text,
            )
            if ids.shape[0] > max_total_len:
                too_long += 1
                continue
            tokenized.append((r, ids, plen, fb_label))
    else:
        for r in rollouts:
            ids, plen = _build_input_ids(tokenizer, r.prompt, r.response)
            if ids.shape[0] > max_total_len:
                too_long += 1
                continue
            tokenized.append((r, ids, plen, ""))
    log.info("After tokenization: %d kept, %d dropped for length > %d",
             len(tokenized), too_long, max_total_len)

    # Sort by length descending for tighter padding
    tokenized.sort(key=lambda t: t[1].shape[0], reverse=True)

    # Output rows
    proj_rows: list[dict[str, Any]] = []
    join_rows: list[dict[str, Any]] = []

    total_batches = (len(tokenized) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(tokenized), desc=f"{eval_name}")
    n_done_since_save = 0

    cache: dict[int, torch.Tensor] = {}
    pre_hooks = [
        (block_modules[L], _make_per_token_pre_hook(L, cache))
        for L in layers_needed
    ]

    for batch_idx in range(total_batches):
        s = batch_idx * batch_size
        e = min(s + batch_size, len(tokenized))
        batch = tokenized[s:e]
        ids_list = [b[1] for b in batch]
        input_ids, attn_mask = _left_pad_stack(ids_list, pad_id)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        cache.clear()
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        # cache[L]: (B, L_pad, d_model) on device, model dtype.
        # Project on device (cast both to f32 to avoid bf16 underflow on large d_model),
        # then move only the (B, L_pad) projections to CPU.
        projs_by_name: dict[str, torch.Tensor] = {}
        for layer, vs_at_layer in v_by_layer.items():
            act = cache[layer].float()  # (B, L_pad, d_model) f32
            for v in vs_at_layer:
                # v_on_device[v.name] is already on device & f32
                p = torch.einsum("bld,d->bl", act, v_on_device[v.name])
                projs_by_name[v.name] = p.cpu()

        # Unpack per-rollout, slice non-pad positions
        L_pad = input_ids.shape[1]
        for b_idx, (r, ids, plen, fb_label) in enumerate(batch):
            actual_len = int(ids.shape[0])
            pad_len = L_pad - actual_len
            # Token strings (decode each id individually)
            id_list = ids.tolist()
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            assistant_mask = [False] * plen + [True] * (actual_len - plen)
            per_vec: dict[str, list[float]] = {}
            for name, p in projs_by_name.items():
                slice_ = p[b_idx, pad_len:].numpy().astype(np.float32)  # (actual_len,)
                assert slice_.shape[0] == actual_len, (slice_.shape, actual_len)
                per_vec[name] = slice_.tolist()
            mask_np = np.array(assistant_mask, dtype=bool)
            # Build a "typical" mask: assistant tokens with |proj| <= threshold
            # for every captured projection AND position >= sink-pos threshold.
            # Tokens with any massive projection (register tokens, sinks) are
            # excluded — they would otherwise dominate the mean.
            typical_mask = mask_np.copy()
            typical_mask &= np.arange(actual_len) >= _SINK_POS_THRESHOLD
            for projs in per_vec.values():
                typical_mask &= np.abs(np.array(projs, dtype=np.float32)) <= _SINK_MAG_THRESHOLD
            agg_row: dict[str, Any] = {
                **r.meta,
                "response_len_tok": actual_len,
                "prompt_len_tok": plen,
                "assistant_len_tok": int(mask_np.sum()),
                "assistant_len_typical": int(typical_mask.sum()),
            }
            if feedback:
                agg_row["feedback"] = fb_label
            for name, projs in per_vec.items():
                arr = np.array(projs, dtype=np.float32)
                stats = _agg(arr, mask_np, typical_mask=typical_mask)
                for k, val in stats.items():
                    agg_row[f"{name}_{k}"] = val
            join_rows.append(agg_row)

            proj_row: dict[str, Any] = {
                **{k: r.meta[k] for k in ("prompt_idx", "rep_idx")},
                "tokens": tokens,
                "token_ids": id_list,
                "assistant_mask": assistant_mask,
                **{name: per_vec[name] for name in projs_by_name},
            }
            if feedback:
                proj_row["feedback"] = fb_label
            proj_rows.append(proj_row)

        n_done_since_save += len(batch)
        pbar.update(len(batch))

        if n_done_since_save >= save_every:
            _flush(proj_rows, join_rows, output_dir, partial=True)
            n_done_since_save = 0

    pbar.close()

    _flush(proj_rows, join_rows, output_dir, partial=False)

    metadata = {
        "eval": eval_name,
        "feedback_mode": feedback,
        "source_json": str(source_json),
        "model_path": str(model_path),
        "base_model_path": base_model_path,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "n_rollouts_loaded": len(rollouts),
        "n_rollouts_kept": len(tokenized),
        "n_rollouts_dropped_too_long": too_long,
        "max_total_len": max_total_len,
        "batch_size": batch_size,
        "vectors": [
            {"name": v.name, "cv_dir": str(v.cv_dir), "layer": v.layer}
            for v in vectors
        ],
        "layers_captured": layers_needed,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Free GPU memory eagerly so callers can chain (model, eval) jobs.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata


def _flush(proj_rows: list[dict], join_rows: list[dict], output_dir: Path, *, partial: bool) -> None:
    """Write parquets. For partial saves we overwrite in place — atomic via
    write-to-tmp + rename."""
    if not proj_rows:
        return
    proj_df = pd.DataFrame(proj_rows)
    join_df = pd.DataFrame(join_rows)
    tag = ".partial" if partial else ""
    for df, name in [(proj_df, "projections"), (join_df, "joined")]:
        out = output_dir / f"{name}{tag}.parquet"
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out)
    if not partial:
        # Drop any leftover .partial files from prior in-progress runs.
        for p in [output_dir / "projections.partial.parquet",
                  output_dir / "joined.partial.parquet"]:
            if p.exists():
                p.unlink()
