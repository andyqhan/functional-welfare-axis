#!/usr/bin/env python3
"""Generate concept vector extraction dataset from maze test set.

Creates trajectories with controlled distributions of final tile types and step counts.

Usage (from project root):
    python -m src.maze.create_concept_vector_dataset --output datasets/concept_vector.parquet

Or with custom parameters:
    python -m src.maze.create_concept_vector_dataset \
        --input datasets/maze_test.parquet \
        --output datasets/concept_vector.parquet \
        --num-rows 15000 \
        --path-fraction 0.50 \
        --lava-fraction 0.40 \
        --goal-fraction 0.10 \
        --min-steps 1 \
        --max-steps 15 \
        --seed 42
"""

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Support both direct execution and module execution
try:
    from .maze import Maze
    from .game import Game, DIRECTIONS
except ImportError:
    from maze import Maze
    from game import Game, DIRECTIONS


# Reverse direction map for backtracking
OPPOSITE_DIR = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

# Short direction names for output
DIR_SHORT = {
    "north": "N",
    "south": "S",
    "east": "E",
    "west": "W",
}


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    target_rows: int = 15000
    path_fraction: float = 0.50
    lava_fraction: float = 0.40
    goal_fraction: float = 0.10
    min_steps: int = 1
    max_steps: int = 15
    max_attempts: int = 1000
    seed: int = 42


def get_adjacent_tiles(maze: Maze, pos: tuple[int, int]) -> dict[str, tuple[str, tuple[int, int]]]:
    """Get adjacent tiles and their positions for a given position.

    Returns dict of direction -> (tile_type, new_position)
    """
    x, y = pos
    result = {}
    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = x + dx, y + dy
        if maze.is_in_bounds(nx, ny):
            tile = maze.get_tile(nx, ny)
            result[direction] = (tile, (nx, ny))
    return result


def random_walk_path(
    maze: Maze,
    target_steps: int,
    max_attempts: int = 1000,
    rng: random.Random = None
) -> tuple[list[str], tuple[int, int]] | None:
    """Generate a random walk of exactly target_steps that stays on PATH tiles.

    Uses backtracking when necessary to achieve exact step count.
    Returns (moves, final_position) or None if failed.
    """
    if rng is None:
        rng = random.Random()

    tc = maze.tile_config

    for _ in range(max_attempts):
        pos = maze.start
        moves = []

        for step in range(target_steps):
            adjacent = get_adjacent_tiles(maze, pos)

            # Filter to PATH-only tiles (avoid LAVA and GOAL)
            valid_dirs = [
                d for d, (tile, _) in adjacent.items()
                if tile == tc.PATH
            ]

            if not valid_dirs:
                # No valid moves - try backtracking if we have moves
                if moves:
                    # Go back the way we came
                    last_dir = moves[-1]
                    opposite = OPPOSITE_DIR[last_dir]
                    if opposite in adjacent and adjacent[opposite][0] == tc.PATH:
                        valid_dirs = [opposite]

                if not valid_dirs:
                    break  # Stuck, try another attempt

            direction = rng.choice(valid_dirs)
            _, new_pos = adjacent[direction]
            moves.append(direction)
            pos = new_pos

        if len(moves) == target_steps:
            # Verify final position is on PATH
            if maze.get_tile(pos[0], pos[1]) == tc.PATH:
                return moves, pos

    return None


def random_walk_to_lava(
    maze: Maze,
    target_steps: int,
    max_attempts: int = 1000,
    rng: random.Random = None
) -> tuple[list[str], tuple[int, int]] | None:
    """Generate a random walk that ends by stepping onto LAVA at exactly target_steps.

    Returns (moves, final_position) or None if failed.
    """
    if rng is None:
        rng = random.Random()

    tc = maze.tile_config

    for _ in range(max_attempts):
        pos = maze.start
        moves = []

        for step in range(target_steps):
            adjacent = get_adjacent_tiles(maze, pos)
            is_last_step = (step == target_steps - 1)

            if is_last_step:
                # Last step: must end on LAVA
                valid_dirs = [
                    d for d, (tile, _) in adjacent.items()
                    if tile == tc.LAVA
                ]
            else:
                # Not last step: stay on PATH (avoid early termination)
                valid_dirs = [
                    d for d, (tile, _) in adjacent.items()
                    if tile == tc.PATH
                ]

            if not valid_dirs:
                break  # Try another attempt

            direction = rng.choice(valid_dirs)
            _, new_pos = adjacent[direction]
            moves.append(direction)
            pos = new_pos

        if len(moves) == target_steps:
            # Verify final position is LAVA
            if maze.get_tile(pos[0], pos[1]) == tc.LAVA:
                return moves, pos

    return None


def random_walk_to_goal(
    maze: Maze,
    target_steps: int,
    max_attempts: int = 1000,
    rng: random.Random = None
) -> tuple[list[str], tuple[int, int]] | None:
    """Generate a random walk that ends by stepping onto GOAL at exactly target_steps.

    Returns (moves, final_position) or None if failed.
    """
    if rng is None:
        rng = random.Random()

    tc = maze.tile_config

    # GOAL requires at least min_path_length steps
    optimal_path = maze.get_optimal_path_length()
    if optimal_path is None or target_steps < optimal_path:
        return None

    # Check parity: can only reach goal in steps with same parity as optimal path
    # (due to backtracking adding 2 steps at a time)
    if (target_steps - optimal_path) % 2 != 0:
        return None

    for _ in range(max_attempts):
        pos = maze.start
        moves = []

        for step in range(target_steps):
            adjacent = get_adjacent_tiles(maze, pos)
            is_last_step = (step == target_steps - 1)

            if is_last_step:
                # Last step: must end on GOAL
                valid_dirs = [
                    d for d, (tile, _) in adjacent.items()
                    if tile == tc.GOAL
                ]
            else:
                # Not last step: stay on PATH (avoid early termination)
                valid_dirs = [
                    d for d, (tile, _) in adjacent.items()
                    if tile == tc.PATH
                ]

            if not valid_dirs:
                break  # Try another attempt

            direction = rng.choice(valid_dirs)
            _, new_pos = adjacent[direction]
            moves.append(direction)
            pos = new_pos

        if len(moves) == target_steps:
            # Verify final position is GOAL
            if maze.get_tile(pos[0], pos[1]) == tc.GOAL:
                return moves, pos

    return None


def build_trajectory(maze: Maze, moves: list[str]) -> list[dict]:
    """Convert a sequence of moves into OpenAI chat format.

    Returns list of {"role": "user/assistant", "content": "..."} messages.
    """
    game = Game(maze)
    messages = []

    for move in moves:
        # Add user prompt (environment state)
        messages.append({
            "role": "user",
            "content": game.get_prompt()
        })

        # Add assistant response (short direction)
        messages.append({
            "role": "assistant",
            "content": DIR_SHORT[move]
        })

        # Execute move
        game.move(move)

    return messages


def analyze_maze_capabilities(
    maze: Maze,
    min_steps: int,
    max_steps: int
) -> dict[int, dict[str, bool]]:
    """Analyze which (step_count, final_tile) combinations are achievable.

    Returns {step: {"PATH": bool, "LAVA": bool, "GOAL": bool}}
    Note: Keys use canonical names ("PATH", "LAVA", "GOAL") regardless of tile mode.
    """
    capabilities = {}
    rng = random.Random(42)  # Fixed seed for analysis

    for steps in range(min_steps, max_steps + 1):
        capabilities[steps] = {
            "PATH": False,
            "LAVA": False,
            "GOAL": False,
        }

        # Check PATH (quick test with limited attempts)
        if random_walk_path(maze, steps, max_attempts=100, rng=rng) is not None:
            capabilities[steps]["PATH"] = True

        # Check LAVA
        if random_walk_to_lava(maze, steps, max_attempts=100, rng=rng) is not None:
            capabilities[steps]["LAVA"] = True

        # Check GOAL
        if random_walk_to_goal(maze, steps, max_attempts=100, rng=rng) is not None:
            capabilities[steps]["GOAL"] = True

    return capabilities


def create_sampling_plan(
    maze_data: list[dict],
    config: TrajectoryConfig
) -> list[tuple[int, int, str]]:
    """Create a balanced sampling plan to achieve target distributions.

    Returns list of (maze_idx, num_steps, final_tile) assignments.
    """
    min_steps = config.min_steps
    max_steps = config.max_steps
    num_step_values = max_steps - min_steps + 1
    rows_per_step = config.target_rows // num_step_values

    # Calculate target counts per step for each tile type
    # GOAL is unreachable in steps 1-2 (min optimal path is 3)
    plan = []

    # Pre-analyze all mazes
    print("  Analyzing maze capabilities...")
    maze_caps = []
    for i, maze_row in enumerate(maze_data):
        maze_str = maze_row["extra_info"]["maze_serialization"]
        maze = Maze.deserialize(maze_str)
        caps = analyze_maze_capabilities(maze, min_steps, max_steps)
        maze_caps.append(caps)
        if (i + 1) % 200 == 0:
            print(f"    Analyzed {i + 1}/{len(maze_data)} mazes")

    # For each step count, distribute assignments across mazes
    for steps in range(min_steps, max_steps + 1):
        # Find mazes capable of each tile type at this step count
        path_mazes = [i for i, caps in enumerate(maze_caps) if caps[steps]["PATH"]]
        lava_mazes = [i for i, caps in enumerate(maze_caps) if caps[steps]["LAVA"]]
        goal_mazes = [i for i, caps in enumerate(maze_caps) if caps[steps]["GOAL"]]

        # Calculate targets for this step
        if steps < 3:
            # GOAL unreachable, redistribute to PATH and LAVA
            path_target = int(rows_per_step * (config.path_fraction / (config.path_fraction + config.lava_fraction)))
            lava_target = rows_per_step - path_target
            goal_target = 0
        else:
            path_target = int(rows_per_step * config.path_fraction)
            lava_target = int(rows_per_step * config.lava_fraction)
            goal_target = rows_per_step - path_target - lava_target

        # Assign PATH trajectories
        if path_mazes:
            for j in range(path_target):
                maze_idx = path_mazes[j % len(path_mazes)]
                plan.append((maze_idx, steps, "PATH"))

        # Assign LAVA trajectories
        if lava_mazes:
            for j in range(lava_target):
                maze_idx = lava_mazes[j % len(lava_mazes)]
                plan.append((maze_idx, steps, "LAVA"))

        # Assign GOAL trajectories
        if goal_mazes and goal_target > 0:
            for j in range(goal_target):
                maze_idx = goal_mazes[j % len(goal_mazes)]
                plan.append((maze_idx, steps, "GOAL"))

    random.shuffle(plan)  # Shuffle to avoid patterns
    return plan


def generate_single_trajectory(
    maze: Maze,
    maze_str: str,
    num_steps: int,
    final_tile: str,
    max_attempts: int,
    rng: random.Random
) -> dict | None:
    """Generate a single trajectory row."""

    if final_tile == "PATH":
        result = random_walk_path(maze, num_steps, max_attempts, rng)
    elif final_tile == "LAVA":
        result = random_walk_to_lava(maze, num_steps, max_attempts, rng)
    elif final_tile == "GOAL":
        result = random_walk_to_goal(maze, num_steps, max_attempts, rng)
    else:
        raise ValueError(f"Unknown final_tile: {final_tile}")

    if result is None:
        return None

    moves, final_pos = result
    trajectory = build_trajectory(maze, moves)

    return {
        "maze_str": maze_str,
        "trajectory": trajectory,
        "num_steps": num_steps,
        "final_tile": final_tile,
    }


def generate_dataset(
    input_path: Path,
    config: TrajectoryConfig
) -> list[dict]:
    """Generate the full concept vector dataset."""

    # Load test mazes
    print(f"Loading mazes from {input_path}...")
    df = pd.read_parquet(input_path)
    maze_data = df.to_dict("records")
    print(f"  Loaded {len(maze_data)} mazes")

    # Create sampling plan
    print("Creating sampling plan...")
    plan = create_sampling_plan(maze_data, config)
    print(f"  Created plan with {len(plan)} assignments")

    # Generate trajectories
    print("Generating trajectories...")
    rng = random.Random(config.seed)
    rows = []
    failed = 0

    # Cache deserialized mazes
    maze_cache = {}

    for i, (maze_idx, num_steps, final_tile) in enumerate(plan):
        # Get or deserialize maze
        if maze_idx not in maze_cache:
            maze_str = maze_data[maze_idx]["extra_info"]["maze_serialization"]
            maze_cache[maze_idx] = (Maze.deserialize(maze_str), maze_str)

        maze, maze_str = maze_cache[maze_idx]

        row = generate_single_trajectory(
            maze, maze_str, num_steps, final_tile, config.max_attempts, rng
        )

        if row is not None:
            rows.append(row)
        else:
            failed += 1

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{len(plan)} trajectories ({failed} failed)")

    print(f"  Total: {len(rows)} successful, {failed} failed")
    return rows


def write_parquet(rows: list[dict], output_path: Path) -> None:
    """Write dataset rows to a parquet file."""
    columns = {
        "maze_str": [r["maze_str"] for r in rows],
        "trajectory": [r["trajectory"] for r in rows],
        "num_steps": [r["num_steps"] for r in rows],
        "final_tile": [r["final_tile"] for r in rows],
    }

    table = pa.table(columns)
    pq.write_table(table, output_path)


def print_statistics(rows: list[dict], config: TrajectoryConfig) -> None:
    """Print distribution statistics for validation."""
    print("\n=== Distribution Validation ===")

    # Final tile distribution
    tile_counts = defaultdict(int)
    for row in rows:
        tile_counts[row["final_tile"]] += 1

    total = len(rows)
    print("\nFinal tile distribution:")
    for tile in ["PATH", "LAVA", "GOAL"]:
        count = tile_counts[tile]
        pct = count / total * 100
        print(f"  {tile}: {count} ({pct:.1f}%)")

    # Step distribution
    step_counts = defaultdict(int)
    for row in rows:
        step_counts[row["num_steps"]] += 1

    print("\nStep distribution:")
    for step in range(config.min_steps, config.max_steps + 1):
        count = step_counts[step]
        pct = count / total * 100
        print(f"  Step {step:2d}: {count:5d} ({pct:.1f}%)")

    # Unique mazes used
    unique_mazes = len(set(r["maze_str"] for r in rows))
    print(f"\nUnique mazes used: {unique_mazes}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate concept vector extraction dataset from maze test set"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/maze_test.parquet"),
        help="Input parquet file with test mazes (default: datasets/maze_test.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/concept_vector.parquet"),
        help="Output parquet file (default: datasets/concept_vector.parquet)",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=15000,
        help="Total number of rows to generate (default: 15000)",
    )
    parser.add_argument(
        "--path-fraction",
        type=float,
        default=0.50,
        help="Fraction of rows ending on PATH (default: 0.50)",
    )
    parser.add_argument(
        "--lava-fraction",
        type=float,
        default=0.40,
        help="Fraction of rows ending on LAVA (default: 0.40)",
    )
    parser.add_argument(
        "--goal-fraction",
        type=float,
        default=0.10,
        help="Fraction of rows ending on GOAL (default: 0.10)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum trajectory length (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum trajectory length (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Max attempts per trajectory before giving up (default: 1000)",
    )

    args = parser.parse_args()

    # Validate fractions
    total_frac = args.path_fraction + args.lava_fraction + args.goal_fraction
    if abs(total_frac - 1.0) > 0.001:
        parser.error(f"Fractions must sum to 1.0, got {total_frac}")

    config = TrajectoryConfig(
        target_rows=args.num_rows,
        path_fraction=args.path_fraction,
        lava_fraction=args.lava_fraction,
        goal_fraction=args.goal_fraction,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    # Set global random seed
    random.seed(config.seed)

    # Generate dataset
    rows = generate_dataset(args.input, config)

    # Write output
    print(f"\nWriting {len(rows)} rows to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, args.output)
    print(f"  Done!")

    # Print statistics
    print_statistics(rows, config)


if __name__ == "__main__":
    main()
