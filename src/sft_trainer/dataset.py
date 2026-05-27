"""SFT dataset generation: mazes + perfect trajectories -> conversations."""

import json
import random
from multiprocessing import Pool
from pathlib import Path

from datasets import Dataset

from src.maze.maze import MazeGenerator
from src.maze.tiles import TileConfig
from src.sft_trainer.config import SFTConfig
from src.sft_trainer.trajectory import (
    TRAJECTORY_ALGORITHMS,
    directions_to_conversation,
)


def _generate_single_example(args: tuple) -> dict | None:
    """Worker function for parallel dataset generation.

    Returns None if the maze is degenerate (e.g., start surrounded by lava).
    """
    idx, seed, config_dict = args
    # Reconstruct config-like access from dict
    maze_size = config_dict["maze_size"]
    goal_lava_ratio = config_dict["goal_lava_ratio"]
    tile_chars = config_dict["tile_chars"]
    max_turns = config_dict["max_turns"]
    max_reachable_goals = config_dict["max_reachable_goals"]
    trajectory_algorithm = config_dict["trajectory_algorithm"]
    shuffle_directions = config_dict["shuffle_directions"]
    relative_directions = config_dict["relative_directions"]

    random.seed(seed)
    tc = TileConfig(
        PATH=tile_chars["path"],
        LAVA=tile_chars["lava"],
        GOAL=tile_chars["goal"],
        PLAYER=tile_chars["player"],
        mode="emoji" if not tile_chars["path"].isascii() else "letters",
    )
    gen = MazeGenerator(size=maze_size, goal_lava_ratio=goal_lava_ratio, tile_config=tc)
    maze = gen.generate()

    # Check for degenerate maze (start surrounded by lava)
    safe_tiles = (tc.PATH, tc.GOAL)
    has_safe_neighbor = any(
        maze.is_in_bounds(maze.start[0] + dx, maze.start[1] + dy)
        and maze.get_tile(maze.start[0] + dx, maze.start[1] + dy) in safe_tiles
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]
    )
    if not has_safe_neighbor:
        return None  # Skip degenerate mazes

    algorithm_fn = TRAJECTORY_ALGORITHMS[trajectory_algorithm]
    directions = algorithm_fn(maze, max_turns, max_reachable_goals)
    messages = directions_to_conversation(
        maze, directions, shuffle_directions, relative_directions
    )

    return {"messages": messages, "seed": seed, "idx": idx}


def generate_dataset(
    config: SFTConfig, split: str, num_examples: int
) -> Dataset:
    """Generate SFT dataset with parallel workers.

    Args:
        config: SFT configuration.
        split: "train" or "val".
        num_examples: Number of examples to generate.

    Returns:
        HuggingFace Dataset with "messages" column.
    """
    seed_offset = 0 if split == "train" else 1_000_000

    config_dict = {
        "maze_size": config.maze_size,
        "goal_lava_ratio": config.goal_lava_ratio,
        "tile_chars": config.tile_chars,
        "max_turns": config.max_turns,
        "max_reachable_goals": config.max_reachable_goals,
        "trajectory_algorithm": config.trajectory_algorithm,
        "shuffle_directions": config.shuffle_directions,
        "relative_directions": config.relative_directions,
    }

    # Generate more than needed to account for degenerate mazes
    oversample = int(num_examples * 1.1) + 100
    args_list = [
        (i, config.base_seed + seed_offset + i, config_dict)
        for i in range(oversample)
    ]

    print(f"Generating {num_examples} {split} examples with {config.dataset_workers} workers...")

    if config.dataset_workers <= 1:
        results = [_generate_single_example(a) for a in args_list]
    else:
        with Pool(config.dataset_workers) as pool:
            results = pool.map(_generate_single_example, args_list)

    # Filter out None (degenerate mazes) and truncate
    examples = [r for r in results if r is not None][:num_examples]
    assert len(examples) == num_examples, (
        f"Could only generate {len(examples)}/{num_examples} valid examples. "
        "Try increasing oversample ratio or reducing maze difficulty."
    )

    print(f"  Generated {len(examples)} valid examples ({split})")
    return Dataset.from_list([{"messages": e["messages"]} for e in examples])


def _cache_path(config: SFTConfig, split: str) -> Path:
    """Path to cached dataset file."""
    d = Path(config.dataset_dir)
    return d / f"{config.trajectory_algorithm}_{config.maze_size}_{split}.jsonl"


def load_or_generate_dataset(
    config: SFTConfig, split: str, num_examples: int
) -> Dataset:
    """Load dataset from cache or generate it."""
    cache = _cache_path(config, split)

    if cache.exists():
        print(f"Loading cached {split} dataset from {cache}")
        with open(cache) as f:
            data = [json.loads(line) for line in f]
        if len(data) >= num_examples:
            return Dataset.from_list(data[:num_examples])
        print(f"  Cache has {len(data)} examples, need {num_examples}. Regenerating.")

    dataset = generate_dataset(config, split, num_examples)

    # Cache to disk
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    print(f"  Cached to {cache}")

    return dataset


if __name__ == "__main__":
    # Quick test: generate a small dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=10)
    parser.add_argument("--num_val", type=int, default=5)
    parser.add_argument("--maze_size", type=int, default=15)
    parser.add_argument("--dataset_workers", type=int, default=1)
    args = parser.parse_args()

    config = SFTConfig(
        num_train=args.num_train,
        num_val=args.num_val,
        maze_size=args.maze_size,
        dataset_workers=args.dataset_workers,
        dataset_dir="datasets/sft_test",
    )

    train_ds = load_or_generate_dataset(config, "train", config.num_train)
    val_ds = load_or_generate_dataset(config, "val", config.num_val)

    print(f"\nTrain: {len(train_ds)} examples")
    print(f"Val: {len(val_ds)} examples")
    print(f"\nSample conversation ({len(train_ds[0]['messages'])} messages):")
    for msg in train_ds[0]["messages"][:6]:
        print(f"  [{msg['role']}] {msg['content'][:80]}...")
