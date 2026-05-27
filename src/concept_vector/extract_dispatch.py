"""Unified concept vector extraction entry point.

Registry-based dispatch: --method selects the extraction strategy, shared
downstream pipeline (precompute, explore, sentiment) consumes the standard
output format.

Usage:
    python -m src.concept_vector.extract_dispatch \
        --method off_policy --checkpoint runs/.../global_step_N --output /tmp/test

    python -m src.concept_vector.extract_dispatch \
        --method on_policy --checkpoint runs/.../global_step_N --output /tmp/test \
        --num-mazes 256 --rollouts-per-maze 4

    python -m src.concept_vector.extract_dispatch \
        --method vector_shift_maze \
        --checkpoint runs/.../global_step_200 \
        --pre-rl-checkpoint runs/.../global_step_0 \
        --output /tmp/test
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from .extraction_output import ConceptResult, write_extraction_outputs


def _portable_checkpoint_path(checkpoint: str | None) -> str | None:
    """Normalize checkpoint path for metadata persistence.

    The project-wide convention for `metadata.checkpoint_path` is the
    filesystem-relative form `runs/<name>/global_step_N`.
    Strip the leading slash on Modal absolute paths; leave anything else
    (already-relative, or genuine local absolute under e.g. /scratch) alone.
    """
    if checkpoint is None:
        return None
    if checkpoint.startswith("/runs/"):
        return checkpoint.lstrip("/")
    return checkpoint


def _resolve_format_kwargs(args: argparse.Namespace, checkpoint_path: str | None) -> dict:
    """Build format_kwargs from CLI --format and --no-chat-template flags.

    Returns a dict to be passed as format_kwargs to get_all_tile_activations
    and ultimately to format_trajectory().
    """
    fmt = getattr(args, "format", "auto")

    if fmt != "auto":
        # Explicit format: pass through directly
        if fmt == "chatml":
            return {}  # empty = default ChatML behavior
        return {"format": fmt}

    # Legacy --no-chat-template flag
    if args.no_chat_template:
        return {"format": "plain"}

    # Auto-detect from checkpoint config
    if checkpoint_path:
        from .model_utils import detect_format
        detected = detect_format(checkpoint_path)
        if detected != "chatml":
            return {"format": detected}

    # Default: ChatML (empty kwargs)
    return {}


# ─── Method: off_policy ──────────────────────────────────────────────────────


def extract_off_policy(args: argparse.Namespace) -> dict[str, ConceptResult]:
    """Random walk trajectories, mean(pos) - mean(neg) per tile type."""
    from .extract_all import (
        generate_trajectories,
        compute_concept_vectors,
        TILE_TYPES,
        CONCEPT_NAMES,
    )
    from .activation_extraction import get_all_tile_activations
    from .model_utils import (
        load_model_with_lora,
        load_base_model,
        resolve_base_model_path,
        detect_use_chat_template,
        DEFAULT_BASE_MODEL,
    )
    from .utils import load_tile_config_from_run_config

    # Load model
    if args.base_model_only:
        model, tokenizer = load_base_model(args.base_model)
        tile_config = load_tile_config_from_run_config(Path(args.checkpoint)) if args.checkpoint else None
    else:
        model, tokenizer = load_model_with_lora(args.checkpoint, args.base_model)
        tile_config = load_tile_config_from_run_config(Path(args.checkpoint))

    if tile_config is None:
        from src.maze.tiles import TileConfig
        tile_config = TileConfig.emoji()

    format_kwargs = _resolve_format_kwargs(args, args.checkpoint)

    # Generate trajectories
    trajectories_by_tile = generate_trajectories(
        tile_config=tile_config,
        n_per_tile=args.n_per_tile,
        maze_size=args.maze_size,
        base_seed=args.seed,
    )

    # Extract activations
    tile_results = get_all_tile_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories_by_tile=trajectories_by_tile,
        batch_size=args.batch_size,
        positions=[-1],
        store_individual=True,
        format_kwargs=format_kwargs,
    )

    # Compute concept vectors
    tile_means = {tile: tile_results[tile]["mean"] for tile in TILE_TYPES}
    concept_vectors = compute_concept_vectors(tile_means)
    tile_counts = {t: len(trajectories_by_tile[t]) for t in TILE_TYPES}

    base_model_used = resolve_base_model_path(args.checkpoint, args.base_model) if args.checkpoint else (args.base_model or DEFAULT_BASE_MODEL)

    concepts = {}
    for concept in CONCEPT_NAMES:
        positive_tile = concept.upper()
        negative_tiles = [t for t in TILE_TYPES if t != positive_tile]

        activations_negative = torch.cat(
            [tile_results[t]["activations"] for t in negative_tiles], dim=0
        ) if tile_results[positive_tile]["activations"] is not None else None

        concepts[concept] = ConceptResult(
            mean_diff=concept_vectors[concept],
            activations_positive=tile_results[positive_tile]["activations"],
            activations_negative=activations_negative,
            metadata={
                "concept": concept,
                "extraction_mode": "off_policy",
                "positive_class": positive_tile,
                "negative_class": negative_tiles,
                "base_model_only": args.base_model_only,
                "checkpoint_path": _portable_checkpoint_path(args.checkpoint),
                "base_model": base_model_used,
                "n_per_tile_requested": args.n_per_tile,
                "n_positive": tile_counts[positive_tile],
                "n_negative": sum(tile_counts[t] for t in negative_tiles),
                "maze_size": args.maze_size,
                "generation_seed": args.seed,
                "batch_size": args.batch_size,
                "tile_config": {
                    "mode": tile_config.mode,
                    "tile_chars": {
                        "path": tile_config.PATH,
                        "lava": tile_config.LAVA,
                        "goal": tile_config.GOAL,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    return concepts


# ─── Method: off_policy_one_turn ─────────────────────────────────────────────


def extract_off_policy_one_turn(args: argparse.Namespace) -> dict[str, ConceptResult]:
    """Exhaustive 48 first-turn trajectories, mean(pos) - mean(neg) per tile type."""
    from .extract_all import (
        generate_one_turn_trajectories,
        compute_concept_vectors,
        TILE_TYPES,
        CONCEPT_NAMES,
    )
    from .activation_extraction import get_all_tile_activations
    from .model_utils import (
        load_model_with_lora,
        load_base_model,
        resolve_base_model_path,
        detect_use_chat_template,
        DEFAULT_BASE_MODEL,
    )
    from .utils import load_tile_config_from_run_config

    # Load model
    if args.base_model_only:
        model, tokenizer = load_base_model(args.base_model)
        tile_config = load_tile_config_from_run_config(Path(args.checkpoint)) if args.checkpoint else None
    else:
        model, tokenizer = load_model_with_lora(args.checkpoint, args.base_model)
        tile_config = load_tile_config_from_run_config(Path(args.checkpoint))

    if tile_config is None:
        from src.maze.tiles import TileConfig
        tile_config = TileConfig.emoji()

    format_kwargs = _resolve_format_kwargs(args, args.checkpoint)

    # Generate one-turn trajectories
    trajectories_by_tile = generate_one_turn_trajectories(
        tile_config=tile_config,
        maze_size=args.maze_size,
        base_seed=args.seed,
    )

    # Extract activations
    tile_results = get_all_tile_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories_by_tile=trajectories_by_tile,
        batch_size=args.batch_size,
        positions=[-1],
        store_individual=True,
        format_kwargs=format_kwargs,
    )

    # Compute concept vectors
    tile_means = {tile: tile_results[tile]["mean"] for tile in TILE_TYPES}
    concept_vectors = compute_concept_vectors(tile_means)
    tile_counts = {t: len(trajectories_by_tile[t]) for t in TILE_TYPES}

    base_model_used = resolve_base_model_path(args.checkpoint, args.base_model) if args.checkpoint else (args.base_model or DEFAULT_BASE_MODEL)

    concepts = {}
    for concept in CONCEPT_NAMES:
        positive_tile = concept.upper()
        negative_tiles = [t for t in TILE_TYPES if t != positive_tile]

        activations_negative = torch.cat(
            [tile_results[t]["activations"] for t in negative_tiles], dim=0
        ) if tile_results[positive_tile]["activations"] is not None else None

        concepts[concept] = ConceptResult(
            mean_diff=concept_vectors[concept],
            activations_positive=tile_results[positive_tile]["activations"],
            activations_negative=activations_negative,
            metadata={
                "concept": concept,
                "extraction_mode": "off_policy_one_turn",
                "positive_class": positive_tile,
                "negative_class": negative_tiles,
                "base_model_only": args.base_model_only,
                "checkpoint_path": _portable_checkpoint_path(args.checkpoint),
                "base_model": base_model_used,
                "n_positive": tile_counts[positive_tile],
                "n_negative": sum(tile_counts[t] for t in negative_tiles),
                "maze_size": args.maze_size,
                "generation_seed": args.seed,
                "batch_size": args.batch_size,
                "tile_config": {
                    "mode": tile_config.mode,
                    "tile_chars": {
                        "path": tile_config.PATH,
                        "lava": tile_config.LAVA,
                        "goal": tile_config.GOAL,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    return concepts


# ─── Method: on_policy ───────────────────────────────────────────────────────


_POSITION_SUBDIRS = {"A": "pre_response", "B": "first_analysis", "C": "first_final"}


def _resolve_on_policy_format(args: argparse.Namespace, checkpoint_path) -> str:
    """Pick a rollout format string from CLI flags / checkpoint config."""
    from .model_utils import detect_format
    fmt = getattr(args, "format", "auto")
    if fmt == "auto":
        return detect_format(str(checkpoint_path))
    return fmt


def _load_on_policy_model(args: argparse.Namespace):
    """Resolve checkpoint + load model for on-policy methods. Returns
    (model, tokenizer, checkpoint_path, base_model, format)."""
    from .model_utils import load_base_model, load_model_with_lora
    from src.maze.inference import load_run_config, resolve_checkpoint_path

    if args.base_model_only:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    run_config = load_run_config(checkpoint_path)
    base_model = getattr(args, "model_override", None) or run_config["base_model"]

    fmt = _resolve_on_policy_format(args, checkpoint_path)

    if args.base_model_only:
        model, tokenizer = load_base_model(base_model)
    else:
        model, tokenizer = load_model_with_lora(str(checkpoint_path), base_model)

    return model, tokenizer, checkpoint_path, base_model, fmt


def _write_on_policy_outputs(
    output_dir: Path,
    multipos: dict,
    single_concept_name: str,
    rollouts_metadata: dict,
    base_model_only: bool,
) -> None:
    """Write per-position extraction outputs for the regular on_policy method
    (single 'reward' concept). Each position becomes a sibling subdir of
    output_dir, each containing its own concepts.txt + concept subdir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pos, payload in multipos.items():
        if payload is None:
            continue
        pos_dir = output_dir / _POSITION_SUBDIRS[pos]
        meta = dict(payload["metadata"])
        meta["base_model_only"] = base_model_only
        concepts = {
            single_concept_name: ConceptResult(
                mean_diff=payload["mean_diff"],
                activations_positive=payload["activations_high"],
                activations_negative=payload["activations_low"],
                metadata=meta,
            ),
        }
        write_extraction_outputs(pos_dir, concepts)
        print(f"[on_policy] wrote {pos_dir}")


def _write_short_circuit_outputs(
    output_dir: Path,
    multipos: dict,
    base_model_only: bool,
) -> None:
    """Write per-position lava/goal/path concept vectors for the short-circuit
    method. Each position becomes a sibling subdir of output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pos, concepts_for_pos in multipos.items():
        if concepts_for_pos is None:
            continue
        pos_dir = output_dir / _POSITION_SUBDIRS[pos]
        concepts = {}
        for name, payload in concepts_for_pos.items():
            meta = dict(payload["metadata"])
            meta["base_model_only"] = base_model_only
            concepts[name] = ConceptResult(
                mean_diff=payload["mean_diff"],
                activations_positive=payload["activations_positive"],
                activations_negative=payload["activations_negative"],
                metadata=meta,
            )
        write_extraction_outputs(pos_dir, concepts)
        print(f"[short_circuit] wrote {pos_dir}")


def extract_on_policy(args: argparse.Namespace) -> dict[str, ConceptResult]:
    """Model rollouts, mean(top_10%) - mean(bottom_10%) by reward.

    Captures activations at three positions per trajectory's last assistant turn:
        A (pre_response), B (first_analysis), C (first_final).
    Writes outputs directly to per-position subdirs of `args.output`.
    """
    from .on_policy_extract import (
        run_rollouts_only,
        save_rollout_cache,
        extract_on_policy_multipos,
    )

    model, tokenizer, checkpoint_path, base_model, fmt = _load_on_policy_model(args)

    rollouts = run_rollouts_only(
        model, tokenizer, checkpoint_path,
        num_mazes=args.num_mazes,
        rollouts_per_maze=args.rollouts_per_maze,
        max_turns=getattr(args, "max_turns", 15),
        temperature=args.temperature,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
        format=fmt,
        goal_lava_ratio=getattr(args, "goal_lava_ratio", None),
        max_new_tokens_harmony=getattr(args, "max_new_tokens_harmony", 64),
    )
    # `run_rollouts_only` records `run_config["base_model"]` in metadata, but
    # we may have rolled out a --model-override checkpoint (e.g. a merged
    # Tinker model). Patch the metadata so downstream consumers (explore.py,
    # webapp) load the same weights we actually used.
    rollouts["metadata"]["base_model"] = base_model

    output_dir = Path(args.output)
    save_rollout_cache(rollouts, output_dir / "rollout_cache")

    multipos = extract_on_policy_multipos(
        model, tokenizer, checkpoint_path,
        rollouts=rollouts,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        batch_size=args.batch_size,
    )
    _write_on_policy_outputs(
        output_dir, multipos, "reward", rollouts["metadata"], args.base_model_only,
    )
    return {}  # outputs already written


def extract_on_policy_short_circuit(args: argparse.Namespace) -> dict[str, ConceptResult]:
    """Filter PATH-only-prefix → first non-PATH (LAVA/GOAL) trajectories.

    Lava vector: positive = LAVA-anchor turns, negative = GOAL-anchor + PATH-prefix turns.
    Goal vector: positive = GOAL-anchor turns, negative = LAVA-anchor + PATH-prefix turns.
    Path vector: positive = PATH-prefix turns of qualifying trajectories,
                 negative = LAVA-anchor + GOAL-anchor turns.

    Captures at three positions (A/B/C). Writes outputs per-position.
    """
    from .on_policy_extract import (
        run_rollouts_only,
        save_rollout_cache,
        extract_on_policy_short_circuit_multipos,
    )

    model, tokenizer, checkpoint_path, base_model, fmt = _load_on_policy_model(args)

    rollouts = run_rollouts_only(
        model, tokenizer, checkpoint_path,
        num_mazes=args.num_mazes,
        rollouts_per_maze=args.rollouts_per_maze,
        max_turns=getattr(args, "max_turns", 15),
        temperature=args.temperature,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
        format=fmt,
        goal_lava_ratio=getattr(args, "goal_lava_ratio", None),
        max_new_tokens_harmony=getattr(args, "max_new_tokens_harmony", 64),
    )
    rollouts["metadata"]["base_model"] = base_model  # see extract_on_policy

    output_dir = Path(args.output)
    save_rollout_cache(rollouts, output_dir / "rollout_cache")

    multipos = extract_on_policy_short_circuit_multipos(
        model, tokenizer, checkpoint_path,
        rollouts=rollouts,
        batch_size=args.batch_size,
    )
    _write_short_circuit_outputs(output_dir, multipos, args.base_model_only)
    return {}  # outputs already written


def extract_on_policy_both(args: argparse.Namespace) -> dict[str, ConceptResult]:
    """Run rollouts ONCE and produce both on_policy and on_policy_short_circuit
    extraction outputs. The output directory layout is:
        <args.output>/on_policy/{pre_response,first_analysis,first_final}/reward/...
        <args.output>/on_policy_short_circuit/{...}/{lava,goal,path}/...
        <args.output>/rollout_cache/{trajectories.jsonl, metadata.json}
    """
    from .on_policy_extract import (
        run_rollouts_only,
        save_rollout_cache,
        extract_on_policy_multipos,
        extract_on_policy_short_circuit_multipos,
    )

    model, tokenizer, checkpoint_path, base_model, fmt = _load_on_policy_model(args)

    rollouts = run_rollouts_only(
        model, tokenizer, checkpoint_path,
        num_mazes=args.num_mazes,
        rollouts_per_maze=args.rollouts_per_maze,
        max_turns=getattr(args, "max_turns", 15),
        temperature=args.temperature,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
        format=fmt,
        goal_lava_ratio=getattr(args, "goal_lava_ratio", None),
        max_new_tokens_harmony=getattr(args, "max_new_tokens_harmony", 64),
    )
    rollouts["metadata"]["base_model"] = base_model  # see extract_on_policy

    base_dir = Path(args.output)
    save_rollout_cache(rollouts, base_dir / "rollout_cache")

    print("\n[on_policy_both] regular on_policy extraction")
    multipos_op = extract_on_policy_multipos(
        model, tokenizer, checkpoint_path,
        rollouts=rollouts,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        batch_size=args.batch_size,
    )
    _write_on_policy_outputs(
        base_dir / "on_policy", multipos_op, "reward",
        rollouts["metadata"], args.base_model_only,
    )

    print("\n[on_policy_both] short-circuit extraction")
    multipos_sc = extract_on_policy_short_circuit_multipos(
        model, tokenizer, checkpoint_path,
        rollouts=rollouts,
        batch_size=args.batch_size,
    )
    _write_short_circuit_outputs(
        base_dir / "on_policy_short_circuit", multipos_sc, args.base_model_only,
    )
    return {}  # outputs already written




# ─── Registry & CLI ──────────────────────────────────────────────────────────

METHODS = {
    "off_policy": extract_off_policy,
    "off_policy_one_turn": extract_off_policy_one_turn,
    "on_policy": extract_on_policy,
    "on_policy_short_circuit": extract_on_policy_short_circuit,
    "on_policy_both": extract_on_policy_both,
}

# Methods that handle their own output writing (multi-subdir layout).
# main() will skip the standard write_extraction_outputs call for these.
_SELF_WRITING_METHODS = {"on_policy", "on_policy_short_circuit", "on_policy_both"}


def main():
    parser = argparse.ArgumentParser(
        description="Unified concept vector extraction dispatcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common args
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()),
                        help="Extraction method")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes")
    parser.add_argument("--base-model-only", action="store_true",
                        help="Load base model without LoRA adapter")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override base model path")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Force use_chat_template=False (legacy; prefer --format plain)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "chatml", "harmony", "plain"],
                        help="Trajectory format: auto (detect from config), chatml (Qwen3), "
                             "harmony (GPT-OSS), plain (base model)")
    parser.add_argument("--seed", type=int, default=474747,
                        help="Base seed")

    # off_policy / off_policy_one_turn / vector_shift_maze
    parser.add_argument("--n-per-tile", type=int, default=5000,
                        help="Trajectories per tile type (off_policy methods)")
    parser.add_argument("--maze-size", type=int, default=100,
                        help="Maze grid size (off_policy methods)")

    # on_policy
    parser.add_argument("--num-mazes", type=int, default=1024,
                        help="Number of mazes (on_policy)")
    parser.add_argument("--rollouts-per-maze", type=int, default=8,
                        help="Rollouts per maze (on_policy)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (on_policy)")
    parser.add_argument("--rollout-batch-size", type=int, default=64,
                        help="Rollout batch size (on_policy)")
    parser.add_argument("--top-pct", type=int, default=10,
                        help="Top percentile (on_policy)")
    parser.add_argument("--bottom-pct", type=int, default=10,
                        help="Bottom percentile (on_policy)")
    parser.add_argument("--goal-lava-ratio", type=float, default=None,
                        help="Override goal-lava ratio (on_policy)")
    parser.add_argument("--max-new-tokens-harmony", type=int, default=64,
                        help="Generation budget per turn for harmony rollouts (on_policy)")
    parser.add_argument("--max-turns", type=int, default=15,
                        help="Max turns per rollout (on_policy)")
    parser.add_argument("--model-override", type=str, default=None,
                        help="Override the base_model field from run config "
                             "(useful for merged-LoRA Tinker checkpoints where "
                             "the runs/ dir holds config but not weights)")

    # controlled
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset path (controlled)")
    parser.add_argument("--num-pairs", type=int, default=1000,
                        help="Number of pairs (controlled)")

    # vector_shift_*
    parser.add_argument("--pre-rl-checkpoint", type=str, default=None,
                        help="Pre-RL checkpoint path (vector_shift methods)")

    args = parser.parse_args()

    # Validate args
    if not args.base_model_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --base-model-only is specified")

    if args.method.startswith("vector_shift") and args.pre_rl_checkpoint is None:
        parser.error("--pre-rl-checkpoint is required for vector_shift methods")

    print(f"=== Concept Vector Extraction: {args.method} ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    if args.method.startswith("vector_shift"):
        print(f"Pre-RL checkpoint: {args.pre_rl_checkpoint}")

    # Dispatch to method
    method_fn = METHODS[args.method]
    concepts = method_fn(args)

    output_dir = Path(args.output)

    if args.method in _SELF_WRITING_METHODS:
        # Method handled its own (multi-subdir) writes
        print(f"\n=== Extraction complete: {args.method} (multi-position output) ===")
        print(f"Output: {output_dir}")
    else:
        concept_names = write_extraction_outputs(output_dir, concepts)
        for name in concept_names:
            concept_dir = output_dir / name
            assert (concept_dir / "mean_diff.pt").exists(), (
                f"Output validation failed: {concept_dir}/mean_diff.pt missing"
            )
            assert (concept_dir / "metadata.json").exists(), (
                f"Output validation failed: {concept_dir}/metadata.json missing"
            )
        print(f"\n=== Extraction complete: {len(concept_names)} concepts ===")
        print(f"Concepts: {concept_names}")
        print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
