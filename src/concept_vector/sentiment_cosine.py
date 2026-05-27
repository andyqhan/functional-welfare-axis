"""
Precompute cosine similarity between maze tile / emoji concept vectors and sentiment vectors.

Produces two JSON artifacts in the CV output directory:
  - sentiment_cosine_maze.json  (CPU-only, maze tile CVs vs sentiment vectors)
  - sentiment_cosine_emoji.json (GPU, emoji LOO CVs vs sentiment vectors)

Usage:
    # Maze tile cosine only (CPU, <1s):
    python -m src.concept_vector.sentiment_cosine \
        --cv-dir artifacts/concept_vectors/run_step100 --maze-only

    # Both maze tile and emoji cosine (GPU required):
    python -m src.concept_vector.sentiment_cosine \
        --cv-dir artifacts/concept_vectors/run_step100 \
        --checkpoint runs/maze-grpo-xxx/global_step_N
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


# Sentiment vector directories keyed by label, grouped by d_model
SENTIMENT_DIRS_BY_DMODEL = {
    2560: {
        "CAD dataset": Path("artifacts/sentiment_vectors"),
        "Prompt-based": Path("artifacts/sentiment_vectors_prompt"),
    },
    2880: {
        "CAD dataset": Path("artifacts/sentiment_vectors_gptoss20b"),
        "Prompt-based": Path("artifacts/sentiment_vectors_prompt_gptoss20b"),
    },
    4096: {
        "CAD dataset": Path("artifacts/sentiment_vectors_8b"),
        "Prompt-based": Path("artifacts/sentiment_vectors_prompt_8b"),
    },
}

# Known pre-RL emoji vector directories
EMOJI_VECTOR_DIRS = [
    Path("artifacts/emoji_vectors"),
    Path("artifacts/emoji_vectors_8b"),
    Path("artifacts/emoji_vectors_8b_thinking"),
]

CONCEPT_ORDER = ["lava", "goal", "path", "reward"]
MODEL_TYPE_PREFIXES = ("rl_steered_baseline", "base_model")


def _load_sentiment_vectors_for_dmodel(d_model: int) -> dict[str, dict] | None:
    """Load sentiment vectors matching d_model.

    Returns dict mapping label -> {"vector": Tensor, "layers": list[int], "dir": str}
    or None if no match.
    """
    dirs = SENTIMENT_DIRS_BY_DMODEL.get(d_model)
    if not dirs:
        return None

    results = {}
    for label, d in dirs.items():
        diff_path = d / "mean_diff.pt"
        meta_path = d / "metadata.json"
        if not diff_path.exists():
            continue
        vec = torch.load(diff_path, map_location="cpu", weights_only=True)
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        target_layers = meta.get("target_layers", [])
        if not target_layers:
            continue
        # Verify d_model matches
        if vec.shape[-1] != d_model:
            continue
        results[label] = {
            "vector": vec,
            "layers": target_layers,
            "dir": str(d),
        }
    return results if results else None


def _detect_d_model(cv_dir: Path) -> int | None:
    """Detect d_model from the first mean_diff.pt found."""
    for p in cv_dir.rglob("mean_diff.pt"):
        t = torch.load(p, map_location="cpu", weights_only=True)
        return t.shape[-1]
    return None


def compute_maze_sentiment_cosine(cv_dir: Path) -> dict | None:
    """Compute cosine similarity between maze tile concept vectors and sentiment vectors.

    Pure CPU, no model loading. Returns dict suitable for JSON serialization, or None
    if no matching sentiment vectors found.
    """
    cv_dir = Path(cv_dir)
    d_model = _detect_d_model(cv_dir)
    if d_model is None:
        print("No mean_diff.pt found in cv_dir, skipping maze cosine")
        return None

    sentiment_vectors = _load_sentiment_vectors_for_dmodel(d_model)
    if sentiment_vectors is None:
        print(f"No sentiment vectors found for d_model={d_model}")
        return None

    # Find all mean_diff.pt, keyed by relative path
    concept_paths: dict[str, Path] = {}
    for p in cv_dir.rglob("mean_diff.pt"):
        rel = str(p.parent.relative_to(cv_dir))
        concept_paths[rel] = p

    if not concept_paths:
        return None

    # Load tensors
    loaded_tensors: dict[str, torch.Tensor] = {}
    for label, path in concept_paths.items():
        loaded_tensors[label] = torch.load(path, map_location="cpu", weights_only=True)

    # Deduplicate: drop rl_steered_baseline/X and base_model/X if identical to X
    to_remove = []
    for label in list(concept_paths.keys()):
        parts = label.split("/", 1)
        if len(parts) == 2 and parts[0] in MODEL_TYPE_PREFIXES:
            trained_label = parts[1]
            if trained_label in loaded_tensors and torch.equal(
                loaded_tensors[label], loaded_tensors[trained_label]
            ):
                to_remove.append(label)
    for label in to_remove:
        del concept_paths[label]
        del loaded_tensors[label]

    # Compute cosine similarities (maze_tile_cosine)
    rows = []
    for cv_label in sorted(concept_paths.keys()):
        cv = loaded_tensors[cv_label].float()
        n_positions, n_layers_full, cv_d_model = cv.shape
        cv_pos0 = cv[0]  # (n_layers_full, d_model)

        for sent_label, sent_info in sentiment_vectors.items():
            sv = sent_info["vector"].float()[0]  # (n_target, d_model)
            if sv.shape[-1] != cv_d_model:
                continue
            target_layers = sent_info["layers"]
            for cache_idx, layer_idx in enumerate(target_layers):
                if layer_idx >= n_layers_full:
                    continue
                sim = F.cosine_similarity(
                    cv_pos0[layer_idx].unsqueeze(0),
                    sv[cache_idx].unsqueeze(0),
                    dim=1,
                ).item()
                rows.append({
                    "concept_vector": cv_label,
                    "sentiment_source": sent_label,
                    "layer": layer_idx,
                    "cosine_similarity": round(sim, 6),
                })

    if not rows:
        return None

    # Compute dumbbell data (pre-RL vs post-RL)
    # Load baseline vectors (inline baselines)
    baseline_vectors: dict[str, torch.Tensor] = {}
    for concept in CONCEPT_ORDER:
        p = cv_dir / "baseline" / concept / "mean_diff.pt"
        if p.exists():
            baseline_vectors[concept] = torch.load(p, map_location="cpu", weights_only=True)

    dumbbell_rows = []
    if baseline_vectors:
        # Get all target layers from the first sentiment source
        first_sent = next(iter(sentiment_vectors.values()))
        target_layers = first_sent["layers"]

        for concept in CONCEPT_ORDER:
            post_rl_path = cv_dir / concept / "mean_diff.pt"
            if not post_rl_path.exists():
                continue
            post_rl = torch.load(post_rl_path, map_location="cpu", weights_only=True).float()
            post_rl_vec = post_rl[0]  # (n_layers, d_model)

            pre_rl = None
            if concept in baseline_vectors:
                pre_rl = baseline_vectors[concept].float()[0]

            for sent_label, sent_info in sentiment_vectors.items():
                sv = sent_info["vector"].float()[0]
                if sv.shape[-1] != post_rl_vec.shape[-1]:
                    continue
                s_layers = sent_info["layers"]
                for cache_idx, layer_idx in enumerate(s_layers):
                    if layer_idx >= post_rl_vec.shape[0]:
                        continue
                    post_sim = F.cosine_similarity(
                        post_rl_vec[layer_idx].unsqueeze(0),
                        sv[cache_idx].unsqueeze(0),
                        dim=1,
                    ).item()
                    pre_sim = None
                    if pre_rl is not None and layer_idx < pre_rl.shape[0] and pre_rl.shape[-1] == sv.shape[-1]:
                        pre_sim = F.cosine_similarity(
                            pre_rl[layer_idx].unsqueeze(0),
                            sv[cache_idx].unsqueeze(0),
                            dim=1,
                        ).item()
                    dumbbell_rows.append({
                        "concept": concept,
                        "sentiment_source": sent_label,
                        "layer": layer_idx,
                        "pre_rl_sim": round(pre_sim, 6) if pre_sim is not None else None,
                        "post_rl_sim": round(post_sim, 6),
                        "delta_sim": round(post_sim - pre_sim, 6) if pre_sim is not None else None,
                    })

    # Collect concepts found
    concepts_found = sorted(set(
        label.split("/")[-1] for label in concept_paths.keys()
        if label.split("/")[-1] in CONCEPT_ORDER
    ))

    # Build sentiment sources metadata
    sent_sources_meta = {label: info["dir"] for label, info in sentiment_vectors.items()}
    target_layers = next(iter(sentiment_vectors.values()))["layers"]

    result = {
        "maze_tile_cosine": rows,
        "dumbbell_data": dumbbell_rows if dumbbell_rows else None,
        "metadata": {
            "d_model": d_model,
            "sentiment_sources": sent_sources_meta,
            "target_layers": target_layers,
            "concepts_found": concepts_found,
            "baseline_available": bool(baseline_vectors),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return result


def _find_pre_rl_emoji_vectors(
    pre_rl_model: str, inject_no_think: bool
) -> Path | None:
    """Find pre-existing pre-RL emoji vectors matching model and think config."""
    for d in EMOJI_VECTOR_DIRS:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        # Check pre_rl_model first, fall back to base_model for older caches
        cached_model = meta.get("pre_rl_model") or meta.get("base_model")
        if cached_model != pre_rl_model:
            continue
        # Match inject_no_think (default False if not present)
        if meta.get("inject_no_think", False) != inject_no_think:
            continue
        # Check that concept vectors exist
        if not (d / "all_concept_vectors.pt").exists():
            continue
        return d
    return None


def _read_tile_config(cv_dir: Path) -> dict | None:
    """Read tile_config from any concept's metadata.json in cv_dir."""
    for concept in CONCEPT_ORDER:
        meta_path = cv_dir / concept / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("tile_config")
    return None


def _read_pre_rl_model(cv_dir: Path) -> str | None:
    """Read the pre-RL model path from any concept's metadata.json.

    Checks 'pre_rl_checkpoint' first (vector_shift methods), then falls back
    to 'base_model' for older extraction metadata.
    """
    for concept in CONCEPT_ORDER:
        meta_path = cv_dir / concept / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("pre_rl_checkpoint") or meta.get("base_model")
    return None


def _read_inject_no_think(cv_dir: Path) -> bool:
    """Read inject_no_think from metadata (for 8B thinking models)."""
    for concept in CONCEPT_ORDER:
        meta_path = cv_dir / concept / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("inject_no_think", False)
    return False


def compute_emoji_sentiment_cosine(
    cv_dir: Path, checkpoint: str, batch_size: int = 64
) -> dict | None:
    """Compute cosine similarity between emoji LOO concept vectors and sentiment vectors.

    Requires GPU. Extracts full-universe LOO emoji vectors for both pre-RL and post-RL
    models, then computes cosine similarity with sentiment vectors for the 3 training emoji.
    """
    from .extract_emoji_vectors import (
        extract_emoji_activations,
        get_all_emoji,
        compute_leave_one_out_vectors,
    )
    from .model_utils import load_base_model, load_model_with_lora

    cv_dir = Path(cv_dir)

    # Step 1: Identify training emoji from tile_config
    tile_config = _read_tile_config(cv_dir)
    if tile_config is None or tile_config.get("mode") != "emoji":
        print("Not an emoji-mode run, skipping emoji cosine")
        return None

    tile_chars = tile_config.get("tile_chars", {})
    training_emoji = {}
    for concept in ["lava", "goal", "path"]:
        char = tile_chars.get(concept)
        if char:
            training_emoji[char] = concept
    if not training_emoji:
        print("No training emoji found in tile_config")
        return None

    print(f"Training emoji: {training_emoji}")

    # Step 2: Get all emoji for LOO
    all_emoji_data = get_all_emoji()
    all_emoji_chars = [e for e, _ in all_emoji_data]
    n_emoji = len(all_emoji_chars)

    # Build index map for training emoji
    emoji_indices = {}
    for i, char in enumerate(all_emoji_chars):
        if char in training_emoji:
            emoji_indices[char] = i
    assert len(emoji_indices) == len(training_emoji), (
        f"Not all training emoji found in emoji universe: "
        f"found {list(emoji_indices.keys())}, expected {list(training_emoji.keys())}"
    )

    # Step 3: Detect model and sentiment vectors
    pre_rl_model = _read_pre_rl_model(cv_dir)
    assert pre_rl_model is not None, "pre_rl_checkpoint not found in metadata"
    inject_no_think = _read_inject_no_think(cv_dir)

    d_model = _detect_d_model(cv_dir)
    assert d_model is not None
    sentiment_vectors = _load_sentiment_vectors_for_dmodel(d_model)
    if sentiment_vectors is None:
        print(f"No sentiment vectors for d_model={d_model}")
        return None

    target_layers = next(iter(sentiment_vectors.values()))["layers"]

    # Step 4: Get or compute pre-RL emoji vectors
    pre_rl_source = None
    pre_rl_dir = _find_pre_rl_emoji_vectors(pre_rl_model, inject_no_think)
    if pre_rl_dir is not None:
        print(f"Found cached pre-RL emoji vectors: {pre_rl_dir}")
        pre_rl_cvs = torch.load(
            pre_rl_dir / "all_concept_vectors.pt", map_location="cpu", weights_only=True
        )
        pre_rl_source = str(pre_rl_dir)
    else:
        print(f"No cached pre-RL emoji vectors for {pre_rl_model}, computing...")
        model, tokenizer = load_base_model(base_model_path=pre_rl_model)
        activations = extract_emoji_activations(
            model, tokenizer, all_emoji_chars,
            batch_size=batch_size, inject_no_think=inject_no_think,
        )
        pre_rl_cvs = compute_leave_one_out_vectors(activations)
        # Cache for future use
        sanitized = pre_rl_model.replace("/", "_")
        suffix = "_thinking" if inject_no_think else ""
        cache_dir = Path(f"artifacts/emoji_vectors_{sanitized}{suffix}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(activations, cache_dir / "all_activations.pt")
        torch.save(pre_rl_cvs, cache_dir / "all_concept_vectors.pt")
        # Save emoji list
        emoji_list_data = [{"emoji": e, "name": n} for e, n in all_emoji_data]
        with open(cache_dir / "emoji_list.json", "w") as f:
            json.dump(emoji_list_data, f, ensure_ascii=False, indent=2)
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump({
                "n_emoji": n_emoji,
                "n_layers": model.config.num_hidden_layers,
                "d_model": model.config.hidden_size,
                "pre_rl_model": pre_rl_model,
                "inject_no_think": inject_no_think,
                "positions": [-1],
                "batch_size": batch_size,
                "prompt_template": "Tell me about {emoji}",
                "activation_shape": list(activations.shape),
                "concept_vector_shape": list(pre_rl_cvs.shape),
            }, f, indent=2, ensure_ascii=False)
        pre_rl_source = str(cache_dir)
        print(f"Cached pre-RL emoji vectors to {cache_dir}")
        # Free model memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Step 5: Compute post-RL emoji vectors
    print(f"Computing post-RL emoji vectors from {checkpoint}...")
    model, tokenizer = load_model_with_lora(checkpoint)
    post_rl_activations = extract_emoji_activations(
        model, tokenizer, all_emoji_chars,
        batch_size=batch_size, inject_no_think=inject_no_think,
    )
    post_rl_cvs = compute_leave_one_out_vectors(post_rl_activations)
    del model, tokenizer
    torch.cuda.empty_cache()

    # Step 6: Compute cosine similarities for training emoji
    pre_rl_data = {}
    post_rl_data = {}

    for emoji_char, concept in training_emoji.items():
        idx = emoji_indices[emoji_char]

        pre_entry = {"concept": concept}
        post_entry = {"concept": concept}

        for sent_label, sent_info in sentiment_vectors.items():
            sv = sent_info["vector"].float()[0]  # (n_target, d_model)
            s_layers = sent_info["layers"]

            pre_layer_sims = {}
            post_layer_sims = {}

            for cache_idx, layer_idx in enumerate(s_layers):
                pre_cv = pre_rl_cvs[idx, 0, layer_idx, :].float()  # (d_model,)
                post_cv = post_rl_cvs[idx, 0, layer_idx, :].float()

                pre_sim = F.cosine_similarity(
                    pre_cv.unsqueeze(0), sv[cache_idx].unsqueeze(0), dim=1
                ).item()
                post_sim = F.cosine_similarity(
                    post_cv.unsqueeze(0), sv[cache_idx].unsqueeze(0), dim=1
                ).item()

                pre_layer_sims[str(layer_idx)] = round(pre_sim, 6)
                post_layer_sims[str(layer_idx)] = round(post_sim, 6)

            pre_entry[sent_label] = pre_layer_sims
            post_entry[sent_label] = post_layer_sims

        pre_rl_data[emoji_char] = pre_entry
        post_rl_data[emoji_char] = post_entry

    result = {
        "pre_rl": {
            "method": "full_universe_loo",
            "n_emoji_universe": n_emoji,
            "source": pre_rl_source,
            "data": pre_rl_data,
        },
        "post_rl": {
            "method": "full_universe_loo",
            "n_emoji_universe": n_emoji,
            "checkpoint": checkpoint,
            "data": post_rl_data,
        },
        "metadata": {
            "training_emoji": training_emoji,
            "d_model": d_model,
            "target_layers": target_layers,
            "pre_rl_model": pre_rl_model,
            "inject_no_think": inject_no_think,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Precompute sentiment cosine similarities for CV pipeline"
    )
    parser.add_argument("--cv-dir", type=str, required=True,
                        help="Concept vector output directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (required for emoji cosine)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for emoji activation extraction")
    parser.add_argument("--maze-only", action="store_true",
                        help="Only compute maze tile cosine (CPU-only, skip emoji)")
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    assert cv_dir.exists(), f"CV dir not found: {cv_dir}"

    # Step 1: Maze tile cosine (always)
    print("=" * 60)
    print("Computing maze tile x sentiment cosine similarities...")
    print("=" * 60)
    t0 = time.time()
    maze_result = compute_maze_sentiment_cosine(cv_dir)
    if maze_result is not None:
        out_path = cv_dir / "sentiment_cosine_maze.json"
        with open(out_path, "w") as f:
            json.dump(maze_result, f, indent=2, ensure_ascii=False)
        n_rows = len(maze_result["maze_tile_cosine"])
        n_dumbbell = len(maze_result["dumbbell_data"]) if maze_result["dumbbell_data"] else 0
        print(f"Saved {out_path}: {n_rows} cosine rows, {n_dumbbell} dumbbell rows ({time.time()-t0:.1f}s)")
    else:
        print("Skipped maze cosine (no matching data)")

    # Step 2: Emoji cosine (unless --maze-only)
    if args.maze_only:
        print("--maze-only: skipping emoji cosine")
        return

    if args.checkpoint is None:
        print("ERROR: --checkpoint required for emoji cosine computation")
        raise SystemExit(1)

    print()
    print("=" * 60)
    print("Computing emoji x sentiment cosine similarities...")
    print("=" * 60)
    t0 = time.time()
    emoji_result = compute_emoji_sentiment_cosine(
        cv_dir, args.checkpoint, batch_size=args.batch_size
    )
    if emoji_result is not None:
        out_path = cv_dir / "sentiment_cosine_emoji.json"
        with open(out_path, "w") as f:
            json.dump(emoji_result, f, indent=2, ensure_ascii=False)
        n_emoji = len(emoji_result["pre_rl"]["data"])
        print(f"Saved {out_path}: {n_emoji} training emoji ({time.time()-t0:.1f}s)")
    else:
        print("Skipped emoji cosine (not an emoji-mode run)")


if __name__ == "__main__":
    main()
