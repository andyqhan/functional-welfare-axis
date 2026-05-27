"""
Unified concept vector extraction for all tile types in a single pass.

This is more efficient than running extract.py three times, as it:
1. Loads the model only once
2. Runs forward passes on all trajectories only once
3. Computes all three concept vectors from the single set of activations

Trajectories are generated dynamically from fresh random mazes with the
correct tile config auto-detected from the training run's config.json.

Speedup: ~3x compared to separate extractions.

Usage:
    python -m src.concept_vector.extract_all \
        --checkpoint runs/maze-grpo-xxx/global_step_N/actor \
        --output artifacts/concept_vectors/step_N

    # From base model only (no LoRA adapter)
    python -m src.concept_vector.extract_all \
        --base-model-only \
        --output artifacts/concept_vectors/baseline
"""

import argparse
import json
import random
import sys
import torch
from pathlib import Path
from datetime import datetime

from .model_utils import load_model_with_lora, load_base_model, resolve_base_model_path, detect_use_chat_template, DEFAULT_BASE_MODEL
from .activation_extraction import get_all_tile_activations
from .utils import load_tile_config_from_run_config

from src.maze.tiles import TileConfig
from src.maze.maze import MazeGenerator
from src.maze.game import Game, DIRECTIONS
from src.maze.create_concept_vector_dataset import (
    random_walk_path,
    random_walk_to_lava,
    random_walk_to_goal,
    build_trajectory,
    DIR_SHORT,
)


def generate_output_dir_name(checkpoint_path: Path) -> str:
    """
    Generate a descriptive output directory name from checkpoint path.

    Examples:
        runs/maze-grpo-128k-706120/global_step_175/actor
            -> maze-grpo-128k-706120_step175
    """
    if checkpoint_path.name == "actor":
        step_dir = checkpoint_path.parent.name
        run_dir = checkpoint_path.parent.parent.name
    else:
        step_dir = checkpoint_path.name
        run_dir = checkpoint_path.parent.name

    step_num = step_dir.replace("global_step_", "").replace("step_", "").replace("step", "")

    if len(run_dir) > 40:
        parts = run_dir.split("-")
        job_id = None
        for part in reversed(parts):
            if part.isdigit() and len(part) >= 5:
                job_id = part
                break

        if job_id:
            prefix_parts = []
            for part in parts:
                if part == job_id:
                    break
                prefix_parts.append(part)
                if len("-".join(prefix_parts)) > 20:
                    break
            run_dir = "-".join(prefix_parts[:3]) + "-" + job_id

    return f"{run_dir}_step{step_num}"


TILE_TYPES = ["LAVA", "GOAL", "PATH"]
CONCEPT_NAMES = ["lava", "goal", "path"]

WALK_FNS = {
    "PATH": random_walk_path,
    "LAVA": random_walk_to_lava,
    "GOAL": random_walk_to_goal,
}


def generate_trajectories(
    tile_config: TileConfig,
    n_per_tile: int = 5000,
    maze_size: int = 100,
    goal_lava_ratio: float = 0.20,
    min_steps: int = 1,
    max_steps: int = 15,
    base_seed: int = 474747,
) -> dict[str, list]:
    """Generate maze trajectories for concept vector extraction.

    Creates fresh random mazes and walks them to produce trajectories ending
    on each tile type (PATH, LAVA, GOAL). Distributes evenly across step counts.

    Args:
        tile_config: Tile configuration (determines maze rendering characters)
        n_per_tile: Number of trajectories per tile type (~5000 each)
        maze_size: Maze grid size (square)
        goal_lava_ratio: Ratio of goals to lava tiles
        min_steps: Minimum trajectory length
        max_steps: Maximum trajectory length
        base_seed: Starting seed for reproducibility

    Returns:
        Dict mapping tile type ("LAVA", "GOAL", "PATH") to list of trajectory
        message lists (OpenAI chat format).
    """
    generator = MazeGenerator(
        size=maze_size, goal_lava_ratio=goal_lava_ratio, tile_config=tile_config
    )
    num_step_values = max_steps - min_steps + 1
    n_per_step = n_per_tile // num_step_values
    remainder = n_per_tile % num_step_values

    trajectories_by_tile = {"LAVA": [], "GOAL": [], "PATH": []}
    seed = base_seed

    for tile_type in TILE_TYPES:
        walk_fn = WALK_FNS[tile_type]

        print(f"  Generating {n_per_tile} {tile_type} trajectories "
              f"({n_per_step}-{n_per_step + 1}/step, {num_step_values} step values)...")

        total_collected = 0
        total_failed_steps = 0

        for step_idx, step_count in enumerate(range(min_steps, max_steps + 1)):
            # Distribute remainder across first few step counts
            target = n_per_step + (1 if step_idx < remainder else 0)
            collected = 0
            attempts = 0
            max_attempts = target * 20  # generous attempt budget

            while collected < target and attempts < max_attempts:
                # Seed global random for maze generation (MazeGenerator uses random module)
                random.seed(seed)
                seed += 1
                maze = generator.generate()

                # Use separate RNG for walk to avoid contaminating maze seeds
                rng = random.Random(seed)
                seed += 1

                result = walk_fn(maze, step_count, max_attempts=100, rng=rng)

                if result is not None:
                    moves, final_pos = result
                    trajectory = build_trajectory(maze, moves)
                    trajectories_by_tile[tile_type].append(trajectory)
                    collected += 1

                attempts += 1

            total_collected += collected
            if collected < target:
                total_failed_steps += 1
                print(f"    Warning: only {collected}/{target} for step={step_count} "
                      f"after {attempts} attempts")

        print(f"    Total: {total_collected} {tile_type} trajectories"
              + (f" ({total_failed_steps} step counts undersampled)" if total_failed_steps else ""))

    return trajectories_by_tile


def generate_one_turn_trajectories(
    tile_config: TileConfig,
    maze_size: int = 100,
    goal_lava_ratio: float = 0.20,
    base_seed: int = 474747,
) -> dict[str, list]:
    """Generate exhaustive one-turn trajectories for concept vector extraction.

    For each of the 12 unique (first_move_direction, landed_tile_type) pairs,
    generates 4 trajectories (one per prefill direction), yielding 48 total
    trajectories grouped by concept (16 per tile type).

    The trajectory structure is:
        1. User msg: initial maze state at start position
        2. Assistant msg: first move direction (N/E/S/W)
        3. User msg: maze state after landing
        4. Assistant msg: prefill direction (N/E/S/W) -- activations extracted here

    Args:
        tile_config: Tile configuration
        maze_size: Maze grid size
        goal_lava_ratio: Ratio of goals to lava tiles
        base_seed: Starting seed for maze generation

    Returns:
        Dict mapping tile type ("LAVA", "GOAL", "PATH") to list of trajectory
        message lists (OpenAI chat format). 16 trajectories per tile type.
    """
    generator = MazeGenerator(
        size=maze_size, goal_lava_ratio=goal_lava_ratio, tile_config=tile_config
    )

    # Build reverse lookup: tile char -> canonical type name
    tile_to_type = {
        tile_config.PATH: "PATH",
        tile_config.LAVA: "LAVA",
        tile_config.GOAL: "GOAL",
    }

    # Track which (direction, tile_type) pairs we've found
    # 4 directions × 3 tile types = 12 pairs needed
    direction_names = list(DIRECTIONS.keys())  # north, south, east, west
    needed_pairs: set[tuple[str, str]] = set()
    for d in direction_names:
        for t in TILE_TYPES:
            needed_pairs.add((d, t))

    # Store one (maze, direction, tile_type, landed_position) per found pair
    found_pairs: dict[tuple[str, str], tuple] = {}

    seed = base_seed
    max_mazes = 10000  # Safety limit

    print(f"  Searching for 12 unique (direction, tile_type) pairs...")
    while needed_pairs and seed - base_seed < max_mazes:
        random.seed(seed)
        seed += 1
        maze = generator.generate()

        game = Game(maze)
        start_pos = game.position

        # Check each direction from start
        for dir_name, (dx, dy) in DIRECTIONS.items():
            nx, ny = start_pos[0] + dx, start_pos[1] + dy
            if not maze.is_in_bounds(nx, ny):
                continue

            tile_char = maze.get_tile(nx, ny)
            tile_type = tile_to_type.get(tile_char)
            if tile_type is None:
                continue

            pair = (dir_name, tile_type)
            if pair in needed_pairs:
                found_pairs[pair] = (maze, dir_name, tile_type, (nx, ny))
                needed_pairs.remove(pair)

    mazes_searched = seed - base_seed
    print(f"    Searched {mazes_searched} mazes, found {len(found_pairs)}/12 pairs")

    assert len(needed_pairs) == 0, (
        f"Could not find all (direction, tile_type) pairs after {mazes_searched} mazes. "
        f"Missing: {needed_pairs}"
    )

    # Build trajectories: for each found pair, create 4 variants (one per prefill direction)
    trajectories_by_tile: dict[str, list] = {"LAVA": [], "GOAL": [], "PATH": []}

    for (dir_name, tile_type), (maze, _, _, landed_pos) in sorted(found_pairs.items()):
        game = Game(maze)

        # 1. Initial prompt (maze state at start)
        initial_prompt = game.get_prompt()

        # 2. Execute first move
        game.move(dir_name)
        assert game.position == landed_pos

        # 3. Post-move prompt (maze state at new position)
        post_move_prompt = game.get_prompt()

        # 4. Create 4 prefill variants
        for prefill_dir in direction_names:
            trajectory = [
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": DIR_SHORT[dir_name]},
                {"role": "user", "content": post_move_prompt},
                {"role": "assistant", "content": DIR_SHORT[prefill_dir]},
            ]
            trajectories_by_tile[tile_type].append(trajectory)

    # Verify counts
    for tile_type in TILE_TYPES:
        count = len(trajectories_by_tile[tile_type])
        assert count == 16, (
            f"Expected 16 trajectories for {tile_type}, got {count}"
        )

    total = sum(len(v) for v in trajectories_by_tile.values())
    assert total == 48, f"Expected 48 total trajectories, got {total}"

    print(f"    Generated {total} trajectories (16 per concept)")

    return trajectories_by_tile


def compute_concept_vectors(tile_means: dict) -> dict:
    """
    Compute all three concept vectors from tile-type means.

    Each concept vector is: mean(positive_class) - mean(negative_classes)

    Args:
        tile_means: Dict mapping tile type to mean activation tensor

    Returns:
        Dict mapping concept name to concept vector tensor
    """
    concept_vectors = {}

    for concept in CONCEPT_NAMES:
        positive_tile = concept.upper()
        negative_tiles = [t for t in TILE_TYPES if t != positive_tile]

        # Compute mean of negative classes
        negative_means = [tile_means[t] for t in negative_tiles]
        mean_negative = sum(negative_means) / len(negative_means)

        # Concept vector = positive - negative
        concept_vectors[concept] = tile_means[positive_tile] - mean_negative

    return concept_vectors


def main(
    checkpoint_path: str,
    output_dir: str = None,
    batch_size: int = 32,
    positions: list = None,
    store_individual: bool = False,
    base_model_only: bool = False,
    base_model_path: str = None,
    n_per_tile: int = 5000,
    maze_size: int = 100,
    base_seed: int = 474747,
    one_turn_only_cv: bool = False,
    no_chat_template: bool = False,
):
    """
    Extract all concept vectors (lava, goal, path) in a single pass.

    Generates trajectories dynamically from fresh random mazes with the correct
    tile config auto-detected from the training run's config.json.

    Args:
        checkpoint_path: Path to checkpoint (e.g., runs/maze-grpo-xxx/global_step_N/actor)
        output_dir: Base output directory (creates lava/, goal/, path/ subdirs)
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1])
        store_individual: If True, also save per-sample activations
        base_model_only: If True, load base model without LoRA adapter
        base_model_path: Override base model path
        n_per_tile: Number of trajectories per tile type
        maze_size: Maze grid size for generated mazes
        base_seed: Starting seed for trajectory generation
        one_turn_only_cv: If True, use exhaustive one-turn prompts (48 total) instead of random walks
    """
    if positions is None:
        positions = [-1]

    # Determine model source and resolve tile config
    if base_model_only:
        if checkpoint_path is not None:
            # Load custom tile config from checkpoint dir's config.json,
            # but still use base model (no LoRA)
            tile_config = load_tile_config_from_run_config(Path(checkpoint_path))
            print(f"Base model mode with custom tile config from {checkpoint_path}")
            print(f"Tile config: mode={tile_config.mode}, "
                  f"PATH={tile_config.PATH!r}, LAVA={tile_config.LAVA!r}, GOAL={tile_config.GOAL!r}")
        else:
            tile_config = TileConfig.emoji()
            print(f"Base model mode: using default emoji tile config")
        checkpoint_path = None
    else:
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint path does not exist: {checkpoint_path}"
        tile_config = load_tile_config_from_run_config(checkpoint_path)
        print(f"Tile config: mode={tile_config.mode}, "
              f"PATH={tile_config.PATH!r}, LAVA={tile_config.LAVA!r}, GOAL={tile_config.GOAL!r}")

    # Auto-generate output directory if not specified
    if output_dir is None:
        if base_model_only:
            output_dir = Path("artifacts/concept_vectors") / "baseline"
        else:
            dir_name = generate_output_dir_name(checkpoint_path)
            output_dir = Path("artifacts/concept_vectors") / dir_name
    else:
        output_dir = Path(output_dir)

    print(f"=== Unified Concept Vector Extraction ===")
    print(f"Concepts: lava, goal, path (all three)")
    print(f"Model: {'BASE MODEL (no LoRA)' if base_model_only else f'LoRA checkpoint: {checkpoint_path}'}")
    print(f"Extraction mode: {'one_turn (exhaustive first-turn prompts)' if one_turn_only_cv else 'random_walk (multi-step trajectories)'}")
    print(f"Dataset: dynamic (fresh random mazes)")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Positions: {positions}")
    print(f"Store individual: {store_individual}")
    if not one_turn_only_cv:
        print(f"Trajectories per tile: {n_per_tile}")
    print(f"Maze size: {maze_size}")
    print(f"Generation seed: {base_seed}")

    # 1. Load model ONCE
    print(f"\n--- Loading model ---")
    if base_model_only:
        model, tokenizer = load_base_model(base_model_path)
        use_chat_template = not no_chat_template  # default True; --no-chat-template forces False
    else:
        model, tokenizer = load_model_with_lora(str(checkpoint_path), base_model_path)
        use_chat_template = False if no_chat_template else detect_use_chat_template(str(checkpoint_path))

    format_kwargs = {} if use_chat_template else {"use_chat_template": False}
    if not use_chat_template:
        print(f"  Detected plain-text format (use_chat_template=False)")

    # 2. Generate trajectories dynamically
    print(f"\n--- Generating trajectories ---")
    if one_turn_only_cv:
        print(f"Using one-turn-only mode (exhaustive first-turn prompts)")
        trajectories_by_tile = generate_one_turn_trajectories(
            tile_config=tile_config,
            maze_size=maze_size,
            goal_lava_ratio=0.20,
            base_seed=base_seed,
        )
    else:
        trajectories_by_tile = generate_trajectories(
            tile_config=tile_config,
            n_per_tile=n_per_tile,
            maze_size=maze_size,
            base_seed=base_seed,
        )

    tile_counts = {t: len(trajectories_by_tile[t]) for t in TILE_TYPES}
    print(f"Trajectory breakdown:")
    for tile_type, count in tile_counts.items():
        print(f"  {tile_type}: {count}")

    # Validate we have samples for each tile type
    for tile_type in TILE_TYPES:
        assert tile_counts[tile_type] > 0, f"No {tile_type} trajectories generated"

    # 3. Extract activations for ALL tile types in a SINGLE pass
    print(f"\n--- Extracting activations (unified) ---")
    tile_results = get_all_tile_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories_by_tile=trajectories_by_tile,
        batch_size=batch_size,
        positions=positions,
        store_individual=store_individual,
        format_kwargs=format_kwargs,
    )

    # 4. Compute concept vectors from tile means
    print(f"\n--- Computing concept vectors ---")
    tile_means = {tile: tile_results[tile]["mean"] for tile in TILE_TYPES}
    concept_vectors = compute_concept_vectors(tile_means)

    # 5. Save outputs for each concept
    print(f"\n--- Saving outputs ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    used_base_model = resolve_base_model_path(str(checkpoint_path), base_model_path) if checkpoint_path else (base_model_path or DEFAULT_BASE_MODEL)

    for concept in CONCEPT_NAMES:
        concept_dir = output_dir / concept
        concept_dir.mkdir(parents=True, exist_ok=True)

        positive_tile = concept.upper()
        negative_tiles = [t for t in TILE_TYPES if t != positive_tile]

        # Get the concept vector and class means
        mean_diff = concept_vectors[concept]
        mean_positive = tile_means[positive_tile]

        # Compute mean of negative classes for saving
        negative_means = [tile_means[t] for t in negative_tiles]
        mean_negative = sum(negative_means) / len(negative_means)

        # Save tensors
        torch.save(mean_diff, concept_dir / "mean_diff.pt")
        torch.save(mean_positive, concept_dir / "mean_lava.pt")  # positive class
        torch.save(mean_negative, concept_dir / "mean_other.pt")  # negative class

        # Save individual activations if requested
        if store_individual:
            activations_positive = tile_results[positive_tile]["activations"]
            # Concatenate negative class activations
            activations_negative = torch.cat(
                [tile_results[t]["activations"] for t in negative_tiles],
                dim=0
            )
            torch.save(activations_positive, concept_dir / "activations_lava.pt")
            torch.save(activations_negative, concept_dir / "activations_other.pt")

        # Save metadata
        metadata = {
            "concept": concept,
            "positive_class": positive_tile,
            "negative_class": negative_tiles,
            "base_model_only": base_model_only,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "base_model": used_base_model,
            "dataset": "dynamic",
            "tile_config": {
                "mode": tile_config.mode,
                "tile_chars": {
                    "path": tile_config.PATH,
                    "lava": tile_config.LAVA,
                    "goal": tile_config.GOAL,
                    "player": tile_config.PLAYER,
                },
            },
            "generation_seed": base_seed,
            "maze_size": maze_size,
            "n_per_tile_requested": n_per_tile,
            "n_positive": tile_counts[positive_tile],
            "n_negative": sum(tile_counts[t] for t in negative_tiles),
            "n_lava": tile_counts["LAVA"],
            "n_path": tile_counts["PATH"],
            "n_goal": tile_counts["GOAL"],
            "positions": positions,
            "extraction_mode": "one_turn" if one_turn_only_cv else "random_walk",
            "extraction_position": "last_token_of_final_move",
            "batch_size": batch_size,
            "store_individual": store_individual,
            "tensor_shape": list(mean_diff.shape),
            "tensor_dtype": str(mean_diff.dtype),
            "timestamp": datetime.now().isoformat(),
            "unified_extraction": True,
        }

        if store_individual:
            metadata["activations_positive_shape"] = list(activations_positive.shape)
            metadata["activations_negative_shape"] = list(activations_negative.shape)

        with open(concept_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Report statistics for this concept
        diff_norms = mean_diff.norm(dim=-1).squeeze()
        print(f"\n  {concept.upper()} concept vector:")
        print(f"    Shape: {mean_diff.shape}")
        print(f"    Layer norm range: [{diff_norms.min():.4f}, {diff_norms.max():.4f}]")
        print(f"    Top 5 layers by norm: {diff_norms.argsort(descending=True)[:5].tolist()}")
        print(f"    Saved to: {concept_dir}")

    print(f"\n=== Unified extraction complete ===")
    print(f"All concept vectors saved to: {output_dir}")
    print(f"  lava/  - LAVA vs (GOAL + PATH)")
    print(f"  goal/  - GOAL vs (LAVA + PATH)")
    print(f"  path/  - PATH vs (LAVA + GOAL)")

    return concept_vectors, tile_results


def cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Extract all concept vectors (lava, goal, path) in a single efficient pass",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory. Required unless --base-model-only is specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory (creates lava/, goal/, path/ subdirs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for forward passes",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[-1],
        help="Token positions to extract activations from",
    )
    parser.add_argument(
        "--no-store-individual",
        action="store_true",
        help="Skip saving per-sample activations (saves disk space)",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Load base model without LoRA adapter (for baseline comparison)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--n-per-tile",
        type=int,
        default=5000,
        help="Number of trajectories per tile type",
    )
    parser.add_argument(
        "--maze-size",
        type=int,
        default=100,
        help="Maze grid size for generated mazes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=474747,
        help="Base seed for trajectory generation",
    )
    parser.add_argument(
        "--one-turn-only-cv",
        action="store_true",
        help="Extract concept vectors from exhaustive set of turn-1 prompts (48 total) "
             "instead of random multi-step walks",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Force use_chat_template=False (for base model runs trained without chat format)",
    )

    # Catch removed --dataset arg with helpful error
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Hidden from help
    )

    args = parser.parse_args()

    # Error if --dataset is passed (removed in favor of dynamic generation)
    if args.dataset is not None:
        print(
            "ERROR: --dataset has been removed. Trajectories are now generated "
            "dynamically with the correct tile config auto-detected from the "
            "training run's config.json. The static parquet dataset is no longer needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate: either checkpoint or base-model-only must be specified
    if not args.base_model_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --base-model-only is specified")

    main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        batch_size=args.batch_size,
        positions=args.positions,
        store_individual=not args.no_store_individual,
        base_model_only=args.base_model_only,
        base_model_path=args.base_model,
        n_per_tile=args.n_per_tile,
        maze_size=args.maze_size,
        base_seed=args.seed,
        one_turn_only_cv=args.one_turn_only_cv,
        no_chat_template=args.no_chat_template,
    )


if __name__ == "__main__":
    cli()
