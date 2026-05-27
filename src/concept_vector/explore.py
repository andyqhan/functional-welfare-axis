"""
Interactive chatbot for exploring concept vector interventions.

Applies ActAdd (adding scaled concept vectors) or ablation (projecting out
the concept direction) to model activations during inference.

Usage:
    # Interactive mode
    python -m src.concept_vector.explore \
        --dir artifacts/concept_vectors/global_step_175 \
        --vector-factors "0 4 -4 abl" \
        --layer 22

    # Batch mode
    # Outputs config_<run_id>_<timestamp>.json and results_<run_id>_<timestamp>.json
    # directly to the concept vector directory (--dir)
    python -m src.concept_vector.explore \
        --dir artifacts/concept_vectors/global_step_175 \
        --vector-factors "0 4 -4 abl" \
        --dataset datasets/concept_vector_eval_prompts.json \
        --run-id 12345 \
        --batch
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log_gpu_memory(label: str = ""):
    """Log GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"  [GPU Memory] {label}: allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB",
            flush=True,
        )

from src.concept_vector.model_utils import (
    load_model_with_lora,
    get_model_block_modules,
    parse_harmony_output,
    DEFAULT_BASE_MODEL,
)


# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chatbot with concept vector interventions"
    )
    parser.add_argument(
        "--dir",
        type=str,
        action="append",
        required=True,
        help="Path to concept vector directory containing mean_diff.pt and metadata.json. "
             "Can be specified multiple times to process multiple directories with a single model load.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path (empty string for base model only)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to use for concept vector. If not specified, auto-selects from metrics.json or defaults to 22.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Space-separated list of layer indices to sweep (e.g. '0 1 2 ... 35'). "
             "When set, the model is loaded once and the alpha-loop runs for each layer, "
             "with per-layer outputs at <output_subdir>/layer_NN/. Mutually exclusive with --layer.",
    )
    parser.add_argument(
        "--vector-factors",
        type=str,
        required=True,
        help='Space-separated list of factors or "abl" (e.g., "4 -1 abl -4")',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode with test prompts from dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/concept_vector_eval_prompts.json",
        help="Path to JSON dataset file with prompts (default: datasets/concept_vector_eval_prompts.json)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier. Used in output filenames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8). Higher values use more VRAM but are faster.",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Skip chat template for base models. Prompts are used directly for completion.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize concept vector to unit norm before steering (default: no normalization)",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=30,
        help="Number of repetitions per prompt (default: 30). Overridden to 1 when temperature=0.",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=0,
        help="Position index into mean_diff's first dimension (default: 0). "
             "E.g., 0 = direction token, 1 = <|im_end|> token for controlled extraction.",
    )
    parser.add_argument(
        "--max-prompts-per-category",
        type=int,
        default=None,
        help="Cap prompts per category (default: None = all). Useful to reduce base model runtime.",
    )
    parser.add_argument(
        "--half-prompts",
        action="store_true",
        help="Take the first half of prompts from each category (floor division).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "chatml", "harmony", "plain"],
        help="Prompt format: auto (detect from metadata), chatml (Qwen3), "
             "harmony (GPT-OSS), plain (no chat template).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode by passing enable_thinking=False to Qwen3 chat template. "
             "Default: use template's built-in default (Qwen3-8B: thinking on; "
             "Qwen3-4B-Instruct-2507: thinking off).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under --dir to write config/results into. "
             "Useful for running multiple token-budget / thinking variants without "
             "overwriting prior results.",
    )
    parser.add_argument(
        "--steer-all-tokens",
        action="store_true",
        help="Legacy behavior: apply steering at every token position "
             "(prompt + generated). Default (off) restricts steering to "
             "non-user-turn positions during prefill so that user-turn tokens "
             "(`<|im_start|>user ... <|im_end|>` for chatml, "
             "`<|start|>user ... <|end|>` for harmony) are left unchanged. "
             "Generated tokens are always steered. Has no effect for "
             "--format plain (no role markers).",
    )
    return parser.parse_args()


def parse_vector_factors(factors_str: str) -> list:
    """Parse space-separated factors string into list of floats or 'abl'."""
    factors = []
    for token in factors_str.split():
        token = token.strip()
        if token.lower() == "abl":
            factors.append("abl")
        else:
            try:
                factors.append(float(token))
            except ValueError:
                raise ValueError(f"Invalid factor: {token}. Expected number or 'abl'.")
    assert len(factors) > 0, "At least one factor required"
    return factors


def format_label(factor) -> str:
    """Format intervention label with color."""
    if factor == "abl":
        return f"{YELLOW}{BOLD}[ABLATION]{RESET}"
    elif factor > 0:
        return f"{GREEN}{BOLD}[+{factor}]{RESET}"
    elif factor < 0:
        return f"{RED}{BOLD}[{factor}]{RESET}"
    else:
        return f"{CYAN}{BOLD}[baseline]{RESET}"


def load_base_model_only(base_model_path: str, torch_dtype=torch.bfloat16):
    """Load base model without LoRA adapter."""
    print(f"Loading base model only: {base_model_path}")
    # See model_utils._resolve_attn_implementation for attention backend selection.
    from src.concept_vector.model_utils import _resolve_attn_implementation
    attn_impl = _resolve_attn_implementation(base_model_path)
    print(f"  attn_implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def make_actadd_hook(cv: torch.Tensor, factor: float, *, steer_mask: torch.Tensor | None = None):
    """Create ActAdd hook that adds scaled concept vector at a single layer.

    If `steer_mask` (shape (B, prompt_len), bool) is given, only positions
    where mask=True are steered during prefill (seq_len > 1). During decode
    (seq_len == 1, KV-cached generation) all positions are steered.

    If `steer_mask` is None, every position is steered (legacy behavior).
    """
    expected_prompt_len = steer_mask.shape[1] if steer_mask is not None else None

    def hook(module, args):
        hidden_states = args[0]  # (B, S, d)
        delta = factor * cv.to(hidden_states.device, hidden_states.dtype)

        if steer_mask is None or hidden_states.shape[1] == 1:
            # Legacy all-positions OR decode (KV-cached, seq_len=1): steer all.
            modified = hidden_states + delta
        else:
            # Prefill with mask. Shape must match the mask we built.
            assert hidden_states.shape[1] == expected_prompt_len, (
                f"steer_mask prompt_len={expected_prompt_len} but hidden_states "
                f"seq_len={hidden_states.shape[1]}"
            )
            assert hidden_states.shape[0] == steer_mask.shape[0], (
                f"steer_mask batch={steer_mask.shape[0]} but hidden_states "
                f"batch={hidden_states.shape[0]}"
            )
            mask = steer_mask.to(hidden_states.device, hidden_states.dtype).unsqueeze(-1)
            modified = hidden_states + mask * delta
        return (modified,) + args[1:]
    return hook


def make_ablation_hook(cv_normalized: torch.Tensor, *, steer_mask: torch.Tensor | None = None):
    """Create ablation hook that projects out concept direction.

    Same `steer_mask` semantics as `make_actadd_hook`: when provided,
    positions where mask=False keep their original residual unchanged
    during prefill; during decode (seq_len=1) all positions are ablated.
    """
    expected_prompt_len = steer_mask.shape[1] if steer_mask is not None else None

    def hook(module, args):
        hidden_states = args[0]  # (B, S, d)
        cv_norm = cv_normalized.to(hidden_states.device, hidden_states.dtype)
        # x' = x - (x·r̂)r̂  →  delta = -(x·r̂)r̂
        proj = torch.einsum("bsd,d->bs", hidden_states, cv_norm)
        delta = -proj.unsqueeze(-1) * cv_norm  # (B, S, d)

        if steer_mask is None or hidden_states.shape[1] == 1:
            modified = hidden_states + delta
        else:
            assert hidden_states.shape[1] == expected_prompt_len, (
                f"steer_mask prompt_len={expected_prompt_len} but hidden_states "
                f"seq_len={hidden_states.shape[1]}"
            )
            assert hidden_states.shape[0] == steer_mask.shape[0], (
                f"steer_mask batch={steer_mask.shape[0]} but hidden_states "
                f"batch={hidden_states.shape[0]}"
            )
            mask = steer_mask.to(hidden_states.device, hidden_states.dtype).unsqueeze(-1)
            modified = hidden_states + mask * delta
        return (modified,) + args[1:]
    return hook


def install_hooks(
    model, layers, cv: torch.Tensor, layer_idx: int, factor,
    *,
    steer_mask: torch.Tensor | None = None,
) -> list:
    """Install intervention hooks and return handles for cleanup.

    Args:
        steer_mask: Optional bool tensor (B, prompt_len). True = steer this
            position during prefill. False = leave the residual unchanged
            (e.g., user-turn tokens). None = legacy all-positions behavior.
            Decode-time (seq_len=1) forwards always steer regardless of mask.
    """
    handles = []

    if factor == "abl":
        # Ablation: apply to ALL layers
        cv_normalized = cv / cv.norm()
        hook = make_ablation_hook(cv_normalized, steer_mask=steer_mask)
        for layer_module in layers:
            h = layer_module.register_forward_pre_hook(hook)
            handles.append(h)
    elif factor == 0:
        # Baseline: no intervention
        pass
    else:
        # ActAdd: apply to single layer
        hook = make_actadd_hook(cv, factor, steer_mask=steer_mask)
        h = layers[layer_idx].register_forward_pre_hook(hook)
        handles.append(h)

    return handles


PLACEHOLDERS = [
    "{LAVA}", "{PATH}", "{GOAL}",
    "{LAVA_ENGLISH}", "{PATH_ENGLISH}", "{GOAL_ENGLISH}",
    "{LAVA_ENGLISH_CAP}", "{PATH_ENGLISH_CAP}", "{GOAL_ENGLISH_CAP}",
]


def _build_substitution_dict(tile_config: dict) -> dict[str, str]:
    """Build placeholder→value mapping from a tile_config dict.

    Args:
        tile_config: Dict with 'tile_mode' and optional 'tile_chars'.

    Returns:
        Dict mapping placeholder strings (e.g. "{LAVA}") to their values.
    """
    from src.maze.tiles import TileConfig

    tc = TileConfig.from_custom_config(tile_config)
    subs = {
        "{LAVA}": tc.LAVA,
        "{PATH}": tc.PATH,
        "{GOAL}": tc.GOAL,
    }
    for tile_type in ("LAVA", "PATH", "GOAL"):
        english = tc.get_english_name(tile_type)
        subs[f"{{{tile_type}_ENGLISH}}"] = english
        subs[f"{{{tile_type}_ENGLISH_CAP}}"] = english.capitalize()
    return subs


def _has_placeholders(text: str) -> bool:
    """Check if text contains any tile placeholders."""
    return any(p in text for p in PLACEHOLDERS)


def load_dataset(dataset_path: str, tile_config: dict | None = None) -> list[dict]:
    """Load prompts from JSON dataset file, substituting tile placeholders.

    Args:
        dataset_path: Path to JSON file containing prompts
        tile_config: Dict with 'tile_mode' (and optional 'tile_chars').
            Used to filter by mode and substitute placeholders.

    Returns:
        List of prompt dicts with placeholders replaced.

    Raises:
        ValueError: If a prompt contains placeholders but tile_config is None.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    tile_mode = tile_config.get("tile_mode") if tile_config else None
    subs = _build_substitution_dict(tile_config) if tile_config else None

    filtered = []
    for item in data:
        assert "prompt" in item, f"Missing 'prompt' field in dataset item: {item}"
        assert "category" in item, f"Missing 'category' field in dataset item: {item}"

        # Filter by tile mode if specified
        if tile_mode is not None:
            prompt_mode = item.get("mode")
            # Include if: no mode specified (mode-agnostic) OR mode matches
            if prompt_mode is not None and prompt_mode != tile_mode:
                continue

        # Check for placeholders and substitute
        prompt_text = item.get("prompt", "")
        original_text = item.get("original_prompt", "")
        needs_sub = _has_placeholders(prompt_text) or _has_placeholders(original_text)

        if needs_sub:
            if subs is None:
                raise ValueError(
                    f"Prompt contains placeholders but no tile_config provided: {prompt_text!r}"
                )
            item = dict(item)  # shallow copy before mutating
            for placeholder, value in subs.items():
                item["prompt"] = item["prompt"].replace(placeholder, value)
                if "original_prompt" in item:
                    item["original_prompt"] = item["original_prompt"].replace(placeholder, value)

        filtered.append(item)

    return filtered


from src.concept_vector.utils import generate_output_suffix, load_tile_config_from_cv_dir


# Re-export under original name for backward compatibility
load_tile_config_from_checkpoint = load_tile_config_from_cv_dir


def auto_select_best_layer(cv_dir: Path) -> int | None:
    """Auto-select best layer from precomputed metrics.json if available.

    Uses the same algorithm as the pipeline script: averages the best layers
    from AUROC, Cohen's d, and overlap metrics.

    Returns:
        Best layer index, or None if metrics.json not found
    """
    import numpy as np

    metrics_path = cv_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        m = json.load(f)

    auroc = np.array(m["auroc"])
    cohens_d = np.array(m["cohens_d"])
    overlap = np.array(m["overlap"])

    # Find best layer for each metric
    best_auroc = np.argsort(auroc)[::-1][0]  # Higher is better
    best_d = np.argsort(np.abs(cohens_d))[::-1][0]  # Higher absolute value is better
    best_sep = np.argsort(overlap)[0]  # Lower is better

    best_layer = int(np.floor(np.mean([best_auroc, best_d, best_sep])))

    print(f"  Auto-selected layer {best_layer} from metrics.json")
    print(f"    AUROC best: layer {best_auroc} ({auroc[best_auroc]:.4f})")
    print(f"    Cohen's d best: layer {best_d} (|d|={np.abs(cohens_d[best_d]):.4f})")
    print(f"    Overlap best: layer {best_sep} ({overlap[best_sep]:.4f})")

    return best_layer


def save_config(output_dir: Path, suffix: str, args, metadata: dict, vector_factors: list, tile_config: dict | None, cv_dir: Path, layer: int, cv_norm: float | None = None) -> Path:
    """Save run configuration to config_<suffix>.json."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "run_id": args.run_id,
        "concept_vector_dir": str(cv_dir),
        "dataset": str(args.dataset),
        "layer": layer,
        "vector_factors": vector_factors,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "checkpoint_path": args.checkpoint_path,
        "base_model": args.base_model,
        "skip_chat_template": args.skip_chat_template,
        "normalize": args.normalize,
        "cv_norm": cv_norm,
        "tile_mode": tile_config.get("tile_mode") if tile_config else None,
        "tile_chars": tile_config.get("tile_chars") if tile_config else None,
        "concept_vector_metadata": metadata,
    }
    config_path = output_dir / f"config_{suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def _find_resume_results(output_dir: Path, run_id: str | None) -> Path | None:
    """If a prior partial/full `results_{run_id}_*.json` exists, return its path
    so we resume into the same filename instead of producing a new timestamped
    sibling. Picks the newest parseable non-empty file.
    """
    if not run_id:
        return None
    candidates = sorted(
        output_dir.glob(f"results_{run_id}_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list) and data:
            return p
    return None


def _populate_from_partial(
    results_map: dict, prompt_data: list, partial: list,
    channels_map: dict | None = None,
) -> set[str]:
    """Copy responses from a partial `results` list into `results_map`.

    If `channels_map` is provided and the partial file has a
    `response_channels` field (harmony-format runs), populate it in parallel
    so resume does not lose per-channel data already on disk.

    Returns the set of factor_keys that are fully populated (all prompts, all
    reps) after the load — those factors can be skipped in the main loop.
    """
    if len(partial) != len(prompt_data):
        print(f"  [partial-load] size mismatch {len(partial)} vs "
              f"{len(prompt_data)}, ignoring partial", flush=True)
        return set()
    per_factor_ok: dict[str, int] = {}
    for prompt_idx, item in enumerate(partial):
        pd = prompt_data[prompt_idx]
        for factor_key, responses in item.get("responses", {}).items():
            if not isinstance(responses, list):
                responses = [responses]
            if factor_key not in results_map[prompt_idx]:
                results_map[prompt_idx][factor_key] = [None] * pd["n_reps"]
            for rep_idx, resp in enumerate(responses[: pd["n_reps"]]):
                results_map[prompt_idx][factor_key][rep_idx] = resp
            if all(r is not None for r in results_map[prompt_idx][factor_key]):
                per_factor_ok[factor_key] = per_factor_ok.get(factor_key, 0) + 1
        if channels_map is not None:
            for factor_key, chans in item.get("response_channels", {}).items():
                if not isinstance(chans, list):
                    chans = [chans]
                if factor_key not in channels_map[prompt_idx]:
                    channels_map[prompt_idx][factor_key] = [None] * pd["n_reps"]
                for rep_idx, ch in enumerate(chans[: pd["n_reps"]]):
                    channels_map[prompt_idx][factor_key][rep_idx] = ch
    return {fk for fk, n in per_factor_ok.items() if n == len(prompt_data)}


def _format_results(prompt_data: list, results_map: dict,
                    vector_factors: list,
                    channels_map: dict | None = None) -> list:
    """Internal results_map → final list-of-dicts. Only emits factors where
    every (prompt, rep) has a non-None response — partial factors are hidden.

    If `channels_map` is provided and populated (harmony-format runs), a
    parallel `response_channels` field is emitted alongside `responses`.
    """
    complete_factors = []
    for factor in vector_factors:
        factor_key = "abl" if factor == "abl" else str(factor)
        all_done = all(
            isinstance(results_map[pi].get(factor_key), list)
            and all(r is not None for r in results_map[pi][factor_key])
            for pi in range(len(prompt_data))
        )
        if all_done:
            complete_factors.append((factor, factor_key))

    def _has_channels(prompt_idx: int, factor_key: str) -> bool:
        if channels_map is None:
            return False
        ch = channels_map[prompt_idx].get(factor_key)
        return isinstance(ch, list) and any(c is not None for c in ch)

    results = []
    for prompt_idx, pd in enumerate(prompt_data):
        prompt_result = {
            "prompt": pd["prompt"],
            "category": pd["category"],
            "n_reps": pd["n_reps"],
            "responses": {},
        }
        if pd.get("original_prompt") is not None:
            prompt_result["original_prompt"] = pd["original_prompt"]
        if pd.get("framing") is not None:
            prompt_result["framing"] = pd["framing"]
        if pd.get("target") is not None:
            prompt_result["target"] = pd["target"]
        if pd.get("metadata") is not None:
            prompt_result["metadata"] = pd["metadata"]
        if any(_has_channels(prompt_idx, fk) for _, fk in complete_factors):
            prompt_result["response_channels"] = {}
        for _factor, factor_key in complete_factors:
            responses = results_map[prompt_idx][factor_key]
            prompt_result["responses"][factor_key] = (
                responses[0] if pd["n_reps"] == 1 else responses
            )
            if "response_channels" in prompt_result and _has_channels(prompt_idx, factor_key):
                chans = channels_map[prompt_idx][factor_key]
                prompt_result["response_channels"][factor_key] = (
                    chans[0] if pd["n_reps"] == 1 else chans
                )
        results.append(prompt_result)
    return results


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically (write to .tmp, then rename). Prevents readers
    from ever seeing a half-written streaming file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_batch_mode(model, tokenizer, layers, cv, layer_idx, vector_factors, args, metadata: dict, tile_config: dict | None, cv_dir: Path, cv_norm: float | None = None):
    """Run test prompts from dataset in batch mode and save results.

    Uses cross-prompt batching: for each factor, batches multiple prompts together
    (with padding) to maximize GPU utilization.

    Streaming-resume: after each factor completes, the current `results` list
    is atomically rewritten to `results_{run_id}_<ts>.json`. On relaunch with
    the same run_id, the prior partial is loaded and completed factors are
    skipped. This bounds preemption loss to ≤ 1 factor of compute.
    """
    # Output goes to the concept vector directory (optionally a subdir for variants)
    output_dir = cv_dir / args.output_subdir if args.output_subdir else cv_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: if a prior partial (or complete) results file with our run_id
    # exists, reuse its filename so the streaming write updates the same path.
    resume_path = _find_resume_results(output_dir, args.run_id)
    if resume_path is not None:
        # Derive suffix from the existing filename: results_<suffix>.json
        suffix = resume_path.stem[len("results_"):]
        print(f"[resume] reusing existing results file: {resume_path.name}",
              flush=True)
    else:
        # Generate filename suffix: <run_id>_<timestamp> or just <timestamp>
        suffix = generate_output_suffix(args.run_id)

    # Load dataset with tile placeholder substitution and mode filtering
    dataset = load_dataset(args.dataset, tile_config=tile_config)
    tile_mode = tile_config.get("tile_mode") if tile_config else None
    print(f"Loaded {len(dataset)} prompts from {args.dataset}")
    if tile_mode:
        print(f"  Filtered to tile_mode={tile_mode!r} (prompts with matching mode or no mode field)")

    # Cap prompts per category if requested
    if args.max_prompts_per_category is not None or args.half_prompts:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for item in dataset:
            by_cat[item["category"]].append(item)
        dataset = []
        for cat, items in sorted(by_cat.items()):
            if args.half_prompts:
                cap = len(items) // 2
            else:
                cap = args.max_prompts_per_category
            dataset.extend(items[:cap])
        label = "half" if args.half_prompts else str(args.max_prompts_per_category)
        print(f"  Capped to {label} per category: {len(dataset)} prompts")

    # Check if temperature is 0 (deterministic mode)
    is_deterministic = args.temperature == 0

    # Save config
    config_path = save_config(output_dir, suffix, args, metadata, vector_factors, tile_config, cv_dir, layer_idx, cv_norm=cv_norm)
    print(f"Saved config to {config_path}")

    print()
    print("=" * 70)
    print(f"{BOLD}BATCH MODE (cross-prompt batching) - {len(dataset)} prompts{RESET}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70, flush=True)
    log_gpu_memory("before tokenization")

    # Step 1: Pre-tokenize all prompts and build generation tasks
    print(f"\n{BOLD}Tokenizing prompts...{RESET}", flush=True)
    prompt_data = []  # List of {prompt, category, n_reps, input_ids, input_len}
    total_generations = 0

    # Use CLI --n-reps, force 1 if deterministic
    n_reps = 1 if is_deterministic else args.n_reps

    for item in dataset:
        prompt = item["prompt"]
        category = item["category"]

        # Tokenize based on resolved format
        if args.resolved_format == "harmony":
            from src.concept_vector.model_utils import format_generation_prompt_harmony
            harmony_str = format_generation_prompt_harmony(
                [{"role": "user", "content": prompt}]
            )
            input_ids = tokenizer(
                harmony_str,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        elif args.resolved_format == "plain" or args.skip_chat_template:
            # Base model: tokenize prompt directly
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
        else:
            # Instruct model: apply chat template (ChatML)
            messages = [{"role": "user", "content": prompt}]
            chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **chat_kwargs,
            )  # Shape: (1, seq_len)

        prompt_data.append({
            "prompt": prompt,
            "category": category,
            "n_reps": n_reps,
            "input_ids": input_ids[0],  # Shape: (seq_len,)
            "input_len": input_ids.shape[1],
            "original_prompt": item.get("original_prompt"),
            "framing": item.get("framing"),
            "target": item.get("target"),
            "metadata": item.get("metadata"),
        })
        total_generations += n_reps * len(vector_factors)

    print(f"  Total generation tasks: {total_generations} ({len(dataset)} prompts × {len(vector_factors)} factors)", flush=True)

    # Initialize results storage
    # results_map[prompt_idx][factor_key] = [response for each rep]
    results_map = {i: {} for i in range(len(prompt_data))}
    # channels_map[prompt_idx][factor_key] = [channels_dict for each rep]
    # Populated only for harmony-format runs; empty otherwise.
    channels_map: dict[int, dict[str, list]] = {
        i: {} for i in range(len(prompt_data))
    }
    is_harmony = args.resolved_format == "harmony"

    # Streaming-resume path (always the same name, whether fresh or resumed)
    results_path = output_dir / f"results_{suffix}.json"

    # If resuming, load prior partial and pre-populate results_map.
    completed_factors: set[str] = set()
    if resume_path is not None:
        try:
            partial = json.loads(resume_path.read_text())
            completed_factors = _populate_from_partial(
                results_map, prompt_data, partial,
                channels_map=channels_map if is_harmony else None,
            )
            if completed_factors:
                print(f"[resume] already-complete factors: "
                      f"{sorted(completed_factors)}", flush=True)
            else:
                print(f"[resume] loaded partial but no factors fully done",
                      flush=True)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[resume] failed to load {resume_path}: {e}; starting fresh",
                  flush=True)

    # Step 3: For each factor, generate all responses with cross-prompt batching
    for factor_idx, factor in enumerate(vector_factors):
        factor_key = "abl" if factor == "abl" else str(factor)

        if factor_key in completed_factors:
            print(f"\n{BOLD}Factor {factor_idx + 1}/{len(vector_factors)}: "
                  f"{factor_key}{RESET} — already complete (resume), skipping",
                  flush=True)
            continue

        # Flatten all tasks for this factor
        all_tasks = []
        for prompt_idx, pdata in enumerate(prompt_data):
            for rep_idx in range(pdata["n_reps"]):
                all_tasks.append((prompt_idx, rep_idx))

        # Length-sort by input_len: puts similar-length prompts in the same
        # batch so padding waste shrinks and the outlier-stall in HF generate
        # clusters at the tail. results_map is keyed by prompt_idx/rep_idx,
        # not by task order, so reordering is safe. Python's sort is stable.
        all_tasks.sort(key=lambda t: prompt_data[t[0]]["input_len"])

        n_batches = (len(all_tasks) + args.batch_size - 1) // args.batch_size
        print(f"\n{BOLD}Factor {factor_idx + 1}/{len(vector_factors)}: {factor_key}{RESET} ({len(all_tasks)} generations, {n_batches} batches)", flush=True)

        # Initialize response storage for this factor
        for prompt_idx in range(len(prompt_data)):
            n_reps_p = prompt_data[prompt_idx]["n_reps"]
            results_map[prompt_idx][factor_key] = [None] * n_reps_p
            if is_harmony:
                channels_map[prompt_idx][factor_key] = [None] * n_reps_p

        # Clear cache before starting this factor to have a clean baseline
        torch.cuda.empty_cache()

        # Process in batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(all_tasks))
            batch_tasks = all_tasks[batch_start:batch_end]
            current_batch_size = len(batch_tasks)

            print(f"  Batch {batch_idx + 1}/{n_batches} ({current_batch_size} sequences)", flush=True)
            if batch_idx == 0:
                log_gpu_memory(f"batch {batch_idx+1} start")

            # Gather input_ids for this batch and find max length
            batch_input_ids = [prompt_data[pid]["input_ids"] for pid, _ in batch_tasks]
            batch_input_lens = [len(ids) for ids in batch_input_ids]
            max_len = max(batch_input_lens)

            # Left-pad to max length (padding on left for causal LM)
            pad_token_id = tokenizer.pad_token_id
            padded_input_ids = []
            attention_masks = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
                    padded = torch.cat([padding, ids])
                    mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)])
                else:
                    padded = ids
                    mask = torch.ones(len(ids), dtype=torch.long)
                padded_input_ids.append(padded)
                attention_masks.append(mask)

            # Stack into batch tensors
            batched_input_ids = torch.stack(padded_input_ids).to(model.device)
            batched_attention_mask = torch.stack(attention_masks).to(model.device)

            # Build per-batch steer mask (assistant-turn-only steering).
            # `--steer-all-tokens` reverts to legacy behavior. "plain" format
            # has no role markers, so mask is always None there.
            steer_mask = None
            if not args.steer_all_tokens and args.resolved_format in ("chatml", "harmony"):
                from src.concept_vector.hook_utils import build_batched_steer_mask
                steer_mask = build_batched_steer_mask(
                    batched_input_ids.cpu(),
                    batch_input_lens,
                    tokenizer,
                    args.resolved_format,
                )

            # Install hooks and generate
            handles = install_hooks(
                model, layers, cv, layer_idx, factor,
                steer_mask=steer_mask,
            )
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        batched_input_ids,
                        attention_mask=batched_attention_mask,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.temperature > 0,
                        pad_token_id=pad_token_id,
                    )

                # Decode each response and store. For harmony-format runs,
                # keep the raw decode (with special tokens) so we can split
                # into per-channel text and expose the analysis channel to
                # downstream judges. The user-facing `response` stays the
                # `final` channel alone for back-compat with existing judges.
                for i, (prompt_idx, rep_idx) in enumerate(batch_tasks):
                    # Response starts after the (padded) input
                    response_ids = output_ids[i][max_len:]
                    if is_harmony:
                        raw = tokenizer.decode(
                            response_ids, skip_special_tokens=False
                        )
                        channels = parse_harmony_output(raw)
                        results_map[prompt_idx][factor_key][rep_idx] = (
                            channels["final"].strip()
                        )
                        channels_map[prompt_idx][factor_key][rep_idx] = channels
                    else:
                        response = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        )
                        results_map[prompt_idx][factor_key][rep_idx] = response.strip()

            finally:
                for h in handles:
                    h.remove()

            # Log memory after first batch and periodically
            if batch_idx == 0:
                log_gpu_memory(f"batch {batch_idx+1} after generate")

            # Clear cache between batches to reduce fragmentation
            if batch_idx % 10 == 9:
                torch.cuda.empty_cache()
                log_gpu_memory(f"batch {batch_idx+1} after empty_cache")

        # Streaming write: after each factor's batches finish, persist the
        # whole-so-far results. Worst-case preemption loss is ≤ 1 factor.
        completed_factors.add(factor_key)
        _atomic_write_json(
            results_path,
            _format_results(
                prompt_data, results_map, vector_factors,
                channels_map=channels_map if is_harmony else None,
            ),
        )
        print(f"  [stream] wrote partial results after factor {factor_key} "
              f"→ {results_path.name}", flush=True)

    # Final memory cleanup and logging
    torch.cuda.empty_cache()
    log_gpu_memory("after all generation complete")

    # Step 4: Format results and do final atomic write.
    # Each per-factor streaming write already used _format_results, so the final
    # write here is idempotent (same content as the last streaming write, with
    # all factors complete).
    print(f"\n{BOLD}Formatting results...{RESET}", flush=True)
    results = _format_results(
        prompt_data, results_map, vector_factors,
        channels_map=channels_map if is_harmony else None,
    )

    # Print sample outputs
    print(f"\n{BOLD}Sample outputs:{RESET}")
    for prompt_idx, pdata in enumerate(prompt_data[:3]):  # Show first 3 prompts
        display_prompt = pdata["prompt"] if len(pdata["prompt"]) <= 80 else pdata["prompt"][:80] + "..."
        print(f"\n{CYAN}[{pdata['category']}] {display_prompt}{RESET}")
        for factor in vector_factors[:3]:  # Show first 3 factors
            factor_key = "abl" if factor == "abl" else str(factor)
            label = format_label(factor)
            responses = results_map[prompt_idx][factor_key]
            if responses and responses[0] is not None:
                resp_preview = responses[0][:100] + "..." if len(responses[0]) > 100 else responses[0]
            else:
                resp_preview = "<none>"
            print(f"  {label} {resp_preview}")

    # Save results (atomic final write)
    _atomic_write_json(results_path, results)

    print()
    print("=" * 70)
    print(f"{BOLD}BATCH MODE COMPLETE{RESET}")
    print(f"Results saved to: {results_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 70)


def main():
    args = parse_args()

    # Convert dir list to Path objects
    cv_dirs = [Path(d) for d in args.dir]

    # FAIL FAST: Validate all directories first
    print("Validating concept vector directories...")
    for cv_dir in cv_dirs:
        assert cv_dir.exists(), f"Concept vector directory not found: {cv_dir}"
        assert (cv_dir / "metadata.json").exists(), f"metadata.json not found in {cv_dir}"
        assert (cv_dir / "mean_diff.pt").exists(), f"mean_diff.pt not found in {cv_dir}"
        print(f"  OK: {cv_dir}")

    # Interactive mode only supports single directory
    if not args.batch and len(cv_dirs) > 1:
        print("ERROR: Interactive mode only supports a single --dir. Use --batch for multiple directories.")
        sys.exit(1)

    # Use first directory to determine model paths (can be overridden by args)
    first_cv_dir = cv_dirs[0]
    with open(first_cv_dir / "metadata.json") as f:
        first_metadata = json.load(f)

    # Resolve prompt format
    if args.format == "auto":
        base_model = first_metadata.get("base_model", "")
        bm_lower = base_model.lower()
        # Tinker-merged checkpoints (e.g. models/tinker-v5-low-160-5677871_step160)
        # are gpt-oss derivatives but the path string contains "tinker", not
        # "gpt-oss". Match either marker — same convention as
        # model_utils._resolve_attn_implementation.
        if "gpt-oss" in bm_lower or "tinker" in bm_lower:
            args.resolved_format = "harmony"
        elif args.skip_chat_template:
            args.resolved_format = "plain"
        else:
            args.resolved_format = "chatml"
    else:
        args.resolved_format = args.format
    print(f"Prompt format: {args.resolved_format}")

    # Resolve model paths
    if args.checkpoint_path is not None:
        # Explicit CLI override always wins
        checkpoint_path = args.checkpoint_path if args.checkpoint_path != "" else None
    elif first_metadata.get("base_model_only"):
        # Extraction was done in base-model-only mode (e.g., merged Tinker checkpoint
        # with no lora_adapter/ subdir) — skip LoRA loading.
        checkpoint_path = None
    else:
        checkpoint_path = first_metadata.get("checkpoint_path")

    base_model_path = args.base_model or first_metadata.get("base_model", DEFAULT_BASE_MODEL)

    # Load model once
    print()
    if checkpoint_path:
        model, tokenizer = load_model_with_lora(checkpoint_path, base_model_path)
    else:
        model, tokenizer = load_base_model_only(base_model_path)

    # Get transformer layers for hook registration
    layers = get_model_block_modules(model)
    print(f"Model has {len(layers)} layers")
    log_gpu_memory("after model load")

    # Parse factors
    vector_factors = parse_vector_factors(args.vector_factors)
    print(f"Vector factors: {vector_factors}")

    # Process each concept vector directory
    for dir_idx, cv_dir in enumerate(cv_dirs):
        if len(cv_dirs) > 1:
            print()
            print("=" * 70)
            print(f"{BOLD}Processing directory {dir_idx + 1}/{len(cv_dirs)}: {cv_dir}{RESET}")
            print("=" * 70)

        # Load metadata for this directory
        with open(cv_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load concept vector tensor
        print(f"Loading concept vector from: {cv_dir}")
        mean_diff = torch.load(cv_dir / "mean_diff.pt", map_location="cpu")
        print(f"  mean_diff shape: {mean_diff.shape}")

        # Determine layer(s): --layers takes precedence; --layer single; else auto-select
        n_layers = mean_diff.shape[1]
        assert not (args.layers and args.layer is not None), \
            "--layer and --layers are mutually exclusive"
        if args.layers:
            layers_to_run = [int(x) for x in args.layers.split()]
            for L in layers_to_run:
                assert 0 <= L < n_layers, f"Layer {L} out of range [0, {n_layers})"
            print(f"  Layer sweep: {len(layers_to_run)} layers {layers_to_run}")
        elif args.layer is None:
            auto_layer = auto_select_best_layer(cv_dir)
            if auto_layer is None:
                auto_layer = min(22, n_layers - 1)
                print(f"  No metrics.json found, using default layer {auto_layer}")
            layers_to_run = [auto_layer]
        else:
            assert 0 <= args.layer < n_layers, f"Layer {args.layer} out of range [0, {n_layers})"
            layers_to_run = [args.layer]

        # Load tile config from checkpoint's config.json (for prompt filtering in batch mode)
        tile_config = load_tile_config_from_checkpoint(cv_dir)
        if tile_config:
            print(f"Detected tile config from training config: tile_mode={tile_config['tile_mode']!r}", end="")
            if tile_config.get("tile_chars"):
                print(f", tile_chars={tile_config['tile_chars']}", end="")
            print()
        else:
            print("No tile config found in training config (will include all prompts)")

        original_output_subdir = args.output_subdir

        for sweep_idx, layer in enumerate(layers_to_run):
            if len(layers_to_run) > 1:
                print()
                print("-" * 70)
                print(f"  Layer sweep step {sweep_idx + 1}/{len(layers_to_run)}: layer={layer}")
                print("-" * 70)
                # Per-layer subdir (preserves any user-supplied output_subdir prefix)
                layer_sub = f"layer_{layer:02d}"
                args.output_subdir = (
                    f"{original_output_subdir.rstrip('/')}/{layer_sub}"
                    if original_output_subdir else layer_sub
                )

            cv = mean_diff[args.position, layer, :]
            cv_norm_val = cv.norm().item()
            print(f"  Using layer {layer}, cv shape: {cv.shape}, norm: {cv_norm_val:.4f}")
            if args.normalize:
                cv = cv / cv.norm()
                print(f"  Normalized cv to unit norm (original norm: {cv_norm_val:.4f})")

            if args.batch:
                run_batch_mode(model, tokenizer, layers, cv, layer, vector_factors, args, metadata, tile_config, cv_dir, cv_norm=cv_norm_val)
            else:
                run_interactive_mode(model, tokenizer, layers, cv, layer, vector_factors, args)

        # Restore for next cv_dir iteration
        args.output_subdir = original_output_subdir

    if len(cv_dirs) > 1:
        print()
        print("=" * 70)
        print(f"{BOLD}ALL DIRECTORIES PROCESSED ({len(cv_dirs)} total){RESET}")
        print("=" * 70)


def run_interactive_mode(model, tokenizer, layers, cv, layer_idx, vector_factors, args):
    """Run interactive chat loop."""
    # Print welcome message
    print()
    print("=" * 60)
    print(f"{BOLD}Concept Vector Explorer (Interactive Mode){RESET}")
    print("=" * 60)
    print(f"Layer: {layer_idx}")
    print(f"Factors: {vector_factors}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Type 'quit', 'exit', or 'q' to stop.")
    print("=" * 60)

    # Chat loop
    messages = []

    while True:
        try:
            user_input = input(f"\n{BOLD}You:{RESET} ").strip()
        except EOFError:
            print("\n[EOF received, exiting]")
            break
        except KeyboardInterrupt:
            print("\n[Interrupted, exiting]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Tokenize conversation
        if args.resolved_format == "harmony":
            from src.concept_vector.model_utils import format_generation_prompt_harmony
            harmony_str = format_generation_prompt_harmony(messages)
            input_ids = tokenizer(
                harmony_str,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(model.device)
        elif args.resolved_format == "plain":
            plain_str = "\n".join(m["content"] for m in messages)
            input_ids = tokenizer(
                plain_str,
                return_tensors="pt",
            ).input_ids.to(model.device)
        else:
            chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **chat_kwargs,
            ).to(model.device)

        # Generate response for each factor
        attention_mask = torch.ones_like(input_ids)

        # Per-call steer mask (assistant-turn-only by default).
        steer_mask = None
        if not args.steer_all_tokens and args.resolved_format in ("chatml", "harmony"):
            from src.concept_vector.hook_utils import build_batched_steer_mask
            steer_mask = build_batched_steer_mask(
                input_ids.cpu(),
                [input_ids.shape[1]],
                tokenizer,
                args.resolved_format,
            )

        for factor in vector_factors:
            handles = install_hooks(
                model, layers, cv, layer_idx, factor,
                steer_mask=steer_mask,
            )
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Decode only the new tokens
                response_ids = output_ids[0][input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                # Print with colored label
                label = format_label(factor)
                print(f"\n{label}")
                print(response.strip())

            finally:
                for h in handles:
                    h.remove()

        # For multi-turn, we don't add the assistant response to history
        # (since we have multiple responses per turn)
        # User can start fresh each turn


if __name__ == "__main__":
    main()
