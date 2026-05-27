"""Trajectory generation algorithms for SFT training data.

Each algorithm takes a maze and produces a list of directions (N/E/S/W) representing
a complete trajectory of exactly max_turns steps.
"""

import random
from collections import deque
from typing import Callable

from src.maze.game import DIRECTIONS, Game
from src.maze.maze import Maze
from src.maze.tiles import TileConfig

# --- Algorithm registry ---

TRAJECTORY_ALGORITHMS: dict[str, Callable] = {}


def register_algorithm(name: str):
    def decorator(fn):
        TRAJECTORY_ALGORITHMS[name] = fn
        return fn
    return decorator


# --- BFS utilities ---

DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W


def bfs_from(maze: Maze, source: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """BFS from source position. Returns {dest: path} where path is the list of
    positions from source to dest (inclusive of both endpoints).

    Only traverses PATH and GOAL tiles (lava is impassable).
    """
    tc = maze.tile_config
    walkable = (tc.PATH, tc.GOAL)

    visited = {source: [source]}
    queue = deque([source])

    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and maze.is_in_bounds(nx, ny):
                tile = maze.get_tile(nx, ny)
                if tile in walkable:
                    visited[(nx, ny)] = visited[(x, y)] + [(nx, ny)]
                    queue.append((nx, ny))

    return visited


def positions_to_directions(positions: list[tuple[int, int]]) -> list[str]:
    """Convert a list of (x, y) positions into direction strings (N/E/S/W)."""
    delta_to_dir = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
    directions = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        directions.append(delta_to_dir[(dx, dy)])
    return directions


# --- Perfect trajectory algorithm ---

def solve_orienteering(dist_matrix: list[list[int]], budget: int) -> tuple[int, list[int]]:
    """Find max-cardinality subset of goals visitable within budget steps.

    Args:
        dist_matrix: (k+1) x (k+1) distance matrix. Index 0 = start, 1..k = goals.
                     dist_matrix[i][j] = BFS shortest path length from node i to node j.
                     Use float('inf') if unreachable.
        budget: maximum total steps.

    Returns:
        (num_goals_visited, visit_order) where visit_order is list of goal indices (0-indexed,
        corresponding to columns 1..k in dist_matrix).
    """
    k = len(dist_matrix) - 1  # number of goals
    if k == 0:
        return 0, []

    INF = float("inf")

    # dp[mask][i] = min steps to visit goals in mask, ending at goal i
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[(-1, -1)] * k for _ in range(1 << k)]

    # Base case: visit just goal j directly from start
    for j in range(k):
        d = dist_matrix[0][j + 1]
        if d <= budget:
            dp[1 << j][j] = d

    # Fill DP
    for mask in range(1, 1 << k):
        for last in range(k):
            if not (mask & (1 << last)):
                continue
            cost_so_far = dp[mask][last]
            if cost_so_far >= INF:
                continue
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                new_cost = cost_so_far + dist_matrix[last + 1][nxt + 1]
                new_mask = mask | (1 << nxt)
                if new_cost <= budget and new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = (mask, last)

    # Find best: mask with max popcount, breaking ties by min steps
    best_count = 0
    best_steps = INF
    best_mask = 0
    best_last = -1
    for mask in range(1, 1 << k):
        pc = bin(mask).count("1")
        if pc < best_count:
            continue
        for i in range(k):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] > budget:
                continue
            if pc > best_count or (pc == best_count and dp[mask][i] < best_steps):
                best_count = pc
                best_steps = dp[mask][i]
                best_mask = mask
                best_last = i

    if best_count == 0:
        return 0, []

    # Reconstruct visit order
    order = []
    mask, last = best_mask, best_last
    while mask > 0 and last >= 0:
        order.append(last)
        prev_mask, prev_last = parent[mask][last]
        mask, last = prev_mask, prev_last
    order.reverse()
    return best_count, order


def pad_trajectory(
    maze: Maze,
    positions: list[tuple[int, int]],
    max_turns: int,
    consumed_goals: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Extend positions list to exactly max_turns+1 positions by wandering on safe tiles.

    The trajectory so far has len(positions) positions = len(positions)-1 steps.
    We need max_turns steps total = max_turns+1 positions.
    """
    tc = maze.tile_config
    safe_tiles = (tc.PATH, tc.GOAL)
    current_positions = list(positions)
    visited_recently = set(positions[-3:]) if len(positions) >= 3 else set(positions)

    while len(current_positions) - 1 < max_turns:
        x, y = current_positions[-1]
        candidates = []
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if maze.is_in_bounds(nx, ny):
                tile = maze.get_tile(nx, ny)
                # Consumed goals appear as PATH to the agent
                if (nx, ny) in consumed_goals:
                    tile = tc.PATH
                if tile in safe_tiles:
                    candidates.append((nx, ny))

        if not candidates:
            # Stuck — start is on an isolated tile surrounded by lava.
            # Fall back to any in-bounds neighbor (may visit lava).
            for dx, dy in DELTAS:
                nx, ny = x + dx, y + dy
                if maze.is_in_bounds(nx, ny):
                    candidates.append((nx, ny))
            if not candidates:
                raise RuntimeError(f"Agent is stuck at {(x, y)} with no neighbors at all")

        # Prefer unvisited tiles
        unvisited = [c for c in candidates if c not in visited_recently]
        if unvisited:
            next_pos = random.choice(unvisited)
        else:
            next_pos = random.choice(candidates)

        current_positions.append(next_pos)
        visited_recently.add(next_pos)
        # Keep visited_recently window small to allow revisiting
        if len(visited_recently) > 6:
            visited_recently = set(current_positions[-3:])

    return current_positions


@register_algorithm("perfect")
def generate_perfect_trajectory(
    maze: Maze,
    max_turns: int = 15,
    max_reachable_goals: int = 15,
) -> list[str]:
    """Generate the highest-reward trajectory for a maze.

    Solves the orienteering problem: find a walk of <= max_turns steps from start
    that visits the maximum number of goals, avoiding lava entirely.
    Pads to exactly max_turns steps by wandering on safe tiles.

    Returns:
        List of exactly max_turns direction strings (N/E/S/W).
    """
    # Phase 1: BFS from start to find reachable goals and their distances
    paths_from_start = bfs_from(maze, maze.start)

    goal_set = set(maze.goals)
    reachable_goals = []
    for goal_pos in maze.goals:
        if goal_pos in paths_from_start:
            dist = len(paths_from_start[goal_pos]) - 1  # path includes start
            if dist <= max_turns:
                reachable_goals.append((dist, goal_pos))

    # Phase 2: Sort by distance and cap
    reachable_goals.sort()
    reachable_goals = reachable_goals[:max_reachable_goals]
    goal_positions = [pos for _, pos in reachable_goals]

    if not goal_positions:
        # No reachable goals — just wander safely
        positions = pad_trajectory(maze, [maze.start], max_turns, set())
        return positions_to_directions(positions)

    # Phase 3: Compute all-pairs shortest paths between start + goals
    nodes = [maze.start] + goal_positions  # index 0 = start, 1..k = goals
    k = len(goal_positions)

    # BFS from each node
    all_paths = {}
    for node in nodes:
        all_paths[node] = bfs_from(maze, node)

    # Build distance matrix
    INF = float("inf")
    dist_matrix = [[INF] * (k + 1) for _ in range(k + 1)]
    path_matrix = [[None] * (k + 1) for _ in range(k + 1)]

    for i in range(k + 1):
        dist_matrix[i][i] = 0
        path_matrix[i][i] = [nodes[i]]
        for j in range(k + 1):
            if i == j:
                continue
            src = nodes[i]
            dst = nodes[j]
            if dst in all_paths[src]:
                bfs_path = all_paths[src][dst]
                dist_matrix[i][j] = len(bfs_path) - 1
                path_matrix[i][j] = bfs_path

    # Phase 4: Solve orienteering problem
    num_goals, visit_order = solve_orienteering(dist_matrix, max_turns)

    if num_goals == 0:
        positions = pad_trajectory(maze, [maze.start], max_turns, set())
        return positions_to_directions(positions)

    # Phase 5: Reconstruct full position path
    positions = [maze.start]
    consumed_goals = set()
    prev_node_idx = 0  # start

    for goal_idx in visit_order:
        node_idx = goal_idx + 1  # +1 because index 0 is start in dist_matrix
        path_segment = path_matrix[prev_node_idx][node_idx]
        assert path_segment is not None, (
            f"No path from {nodes[prev_node_idx]} to {nodes[node_idx]}"
        )
        # Skip first position (already in positions list)
        positions.extend(path_segment[1:])
        consumed_goals.add(nodes[node_idx])
        prev_node_idx = node_idx

    assert len(positions) - 1 <= max_turns, (
        f"Path uses {len(positions) - 1} steps but budget is {max_turns}"
    )

    # Phase 6: Pad to exactly max_turns steps
    if len(positions) - 1 < max_turns:
        positions = pad_trajectory(maze, positions, max_turns, consumed_goals)

    assert len(positions) - 1 == max_turns, (
        f"Expected {max_turns} steps, got {len(positions) - 1}"
    )

    return positions_to_directions(positions)


# --- Conversation generation ---

def directions_to_conversation(
    maze: Maze,
    directions: list[str],
    shuffle_directions: bool = True,
    relative_directions: bool = True,
) -> list[dict[str, str]]:
    """Play a list of directions through a Game and record the multi-turn conversation.

    Returns:
        List of 2*len(directions) messages: alternating user/assistant.
    """
    direction_order = ["north", "east", "south", "west"]
    if shuffle_directions:
        random.shuffle(direction_order)

    game = Game(
        maze,
        wind_frequency=0.0,
        melting_path=False,
        direction_order=direction_order,
        relative_directions=relative_directions,
    )

    messages = []
    messages.append({"role": "user", "content": game.get_prompt()})

    for d in directions:
        messages.append({"role": "assistant", "content": d})
        game.move(d.lower())
        messages.append({"role": "user", "content": game.get_prompt()})

    # Remove trailing user message (no assistant response for it)
    messages = messages[:-1]

    assert len(messages) == 2 * len(directions), (
        f"Expected {2 * len(directions)} messages, got {len(messages)}"
    )
    return messages
