from dataclasses import dataclass, field
from collections import deque
import random

# Support both direct execution and module execution
try:
    from .tiles import TileConfig, DEFAULT_TILE_CONFIG
except ImportError:
    from tiles import TileConfig, DEFAULT_TILE_CONFIG

# Tile types - backward compatibility constants (emoji mode)
PATH = "🍩"
LAVA = "🧁"
GOAL = "🍨"

DEFAULT_SIZE = 100

# Header prefixes for tile config in serialized mazes
TILE_MODE_HEADER_PREFIX = "#tile_mode:"  # Legacy format (backward compat)
TILE_CONFIG_HEADER_PREFIX = "#tile_config:"  # New format with explicit char mapping


@dataclass
class Maze:
    grid: list[list[str]]
    start: tuple[int, int]
    goals: list[tuple[int, int]]
    tile_config: TileConfig = field(default_factory=TileConfig.emoji)

    @property
    def size(self) -> int:
        """Return the size of the maze (assumes square grid)."""
        return len(self.grid)

    def get_tile(self, x: int, y: int) -> str:
        return self.grid[y][x]

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def get_lava_positions(self) -> set[tuple[int, int]]:
        """Return all lava tile positions."""
        positions = set()
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile == self.tile_config.LAVA:
                    positions.add((x, y))
        return positions

    def get_goal_positions(self) -> set[tuple[int, int]]:
        """Return all goal tile positions."""
        return set(self.goals)

    def get_path_count(self) -> int:
        """Return count of walkable path tiles (excluding goals)."""
        count = 0
        for row in self.grid:
            for tile in row:
                if tile == self.tile_config.PATH:
                    count += 1
        return count

    def get_optimal_path_length(self) -> int | None:
        """Return the shortest path length from start to any goal using BFS.

        Returns:
            Number of steps to reach nearest goal, or None if no goal is reachable.
        """
        if not self.goals:
            return None

        goal_set = set(self.goals)
        queue = deque([(self.start, 0)])
        visited = {self.start}
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        while queue:
            (x, y), steps = queue.popleft()

            # Check if we reached a goal
            if (x, y) in goal_set:
                return steps

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and self.is_in_bounds(nx, ny):
                    tile = self.get_tile(nx, ny)
                    # Can walk on PATH or GOAL tiles
                    if tile in (self.tile_config.PATH, self.tile_config.GOAL):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), steps + 1))

        return None  # No goal reachable

    def __str__(self) -> str:
        lines = []
        for row in self.grid:
            lines.append("".join(row))
        return "\n".join(lines)

    def serialize(self) -> str:
        """Serialize the maze to a string representation.

        Includes a header line with tile config for auto-detection during deserialization.
        Format: #tile_config:PATH=A,LAVA=B,GOAL=C,PLAYER=@\\n{grid}
        """
        tc = self.tile_config
        header = f"{TILE_CONFIG_HEADER_PREFIX}PATH={tc.PATH},LAVA={tc.LAVA},GOAL={tc.GOAL},PLAYER={tc.PLAYER}"
        return f"{header}\n{str(self)}"

    @staticmethod
    def _find_center_start(grid: list[list[str]], tile_config: TileConfig) -> tuple[int, int]:
        """Find the nearest PATH tile to the center using BFS.

        Args:
            grid: The maze grid
            tile_config: Tile configuration for determining PATH tiles

        Returns:
            The (x, y) coordinates of the nearest PATH tile to center
        """
        size = len(grid)
        center = (size // 2, size // 2)

        # BFS from center to find nearest PATH tile
        queue = deque([center])
        visited = {center}
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        while queue:
            x, y = queue.popleft()
            tile = grid[y][x]
            if tile == tile_config.PATH:
                return (x, y)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < size and 0 <= ny < size:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # Fallback: return center if no PATH found (shouldn't happen with valid mazes)
        return center

    @classmethod
    def deserialize(cls, s: str, start: tuple[int, int] | None = None) -> "Maze":
        """Deserialize a maze from string representation.

        Auto-detects tile config from header if present, otherwise defaults to emoji mode.
        Supports both new #tile_config: and legacy #tile_mode: header formats.

        Args:
            s: The serialized maze string (optionally with header)
            start: Optional start position. If None, auto-computes center-nearest PATH tile.

        Returns:
            A Maze instance
        """
        lines = s.strip().split("\n")

        # Check for tile config headers (new format first, then legacy)
        tile_config = DEFAULT_TILE_CONFIG  # Default to emoji mode
        if lines and lines[0].startswith(TILE_CONFIG_HEADER_PREFIX):
            # New format: #tile_config:PATH=A,LAVA=B,GOAL=C,PLAYER=@
            config_str = lines[0][len(TILE_CONFIG_HEADER_PREFIX):]
            char_map = {}
            for pair in config_str.split(","):
                key, value = pair.split("=", 1)
                char_map[key.strip()] = value.strip()
            tile_config = TileConfig(
                PATH=char_map["PATH"],
                LAVA=char_map["LAVA"],
                GOAL=char_map["GOAL"],
                PLAYER=char_map["PLAYER"],
                mode="custom",
            )
            lines = lines[1:]
        elif lines and lines[0].startswith(TILE_MODE_HEADER_PREFIX):
            # Legacy format: #tile_mode:letters
            mode = lines[0][len(TILE_MODE_HEADER_PREFIX):]
            tile_config = TileConfig.from_mode(mode)
            lines = lines[1:]

        grid = [list(line) for line in lines]

        # Find ALL goal tiles using the detected tile config
        goals = []
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile == tile_config.GOAL:
                    goals.append((x, y))

        # If start not provided, find center-nearest PATH tile
        if start is None:
            start = cls._find_center_start(grid, tile_config)

        return cls(grid=grid, start=start, goals=goals, tile_config=tile_config)


class MazeGenerator:
    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        goal_lava_ratio: float = 0.20,
        tile_config: TileConfig | None = None,
    ):
        """Initialize the maze generator.

        Args:
            size: Size of the maze grid (square)
            goal_lava_ratio: Ratio of goals to lava tiles (e.g., 0.20 = 1 goal per 5 lava tiles)
            tile_config: Tile configuration (emoji or letters mode). Defaults to emoji.
        """
        self.size = size
        self.goal_lava_ratio = goal_lava_ratio
        self.tile_config = tile_config or DEFAULT_TILE_CONFIG

    def generate(self) -> Maze:
        """Generate a maze with multiple goal tiles.

        The generation algorithm:
        1. Initialize grid with all LAVA
        2. Random walk from center to create initial PATH tiles
        3. Fill remaining interior to meet 10-50% lava constraint
        4. Place goals: num_goals = max(1, int(lava_count * goal_lava_ratio))
        5. Compute start position as center-nearest PATH tile
        """
        size = self.size
        tc = self.tile_config

        # Initialize grid with all lava
        grid = [[tc.LAVA for _ in range(size)] for _ in range(size)]

        # Start random walk from random interior position
        x = random.randint(1, size - 2)
        y = random.randint(1, size - 2)
        grid[y][x] = tc.PATH

        # Random walk length scales with maze size to create interesting paths
        # For 100x100 maze: 500-1500 steps creates organic path structures
        interior_size = size - 2
        walk_length = random.randint(interior_size * 5, interior_size * 15)
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W

        for _ in range(walk_length):
            # Filter valid directions (stay in interior)
            valid_dirs = [
                (dx, dy)
                for dx, dy in directions
                if 1 <= x + dx < size - 1 and 1 <= y + dy < size - 1
            ]
            if not valid_dirs:
                break
            dx, dy = random.choice(valid_dirs)
            x, y = x + dx, y + dy
            grid[y][x] = tc.PATH

        # Fill remaining interior to meet lava percentage constraint (10-50%)
        interior_lava_cells = []
        for iy in range(1, size - 1):
            for ix in range(1, size - 1):
                if grid[iy][ix] == tc.LAVA:
                    interior_lava_cells.append((ix, iy))

        total_interior = (size - 2) ** 2
        min_lava = int(total_interior * 0.10) + 1  # >10%
        max_lava = int(total_interior * 0.50) - 1  # <50%

        current_lava = len(interior_lava_cells)

        # Convert some lava to path if needed to meet constraint
        if current_lava > max_lava:
            to_convert = current_lava - random.randint(min_lava, max_lava)
            random.shuffle(interior_lava_cells)
            for i in range(min(to_convert, len(interior_lava_cells))):
                ix, iy = interior_lava_cells[i]
                grid[iy][ix] = tc.PATH

        # Count final lava tiles in interior
        lava_count = sum(
            1
            for iy in range(1, size - 1)
            for ix in range(1, size - 1)
            if grid[iy][ix] == tc.LAVA
        )

        # Calculate number of goals based on lava count
        num_goals = max(1, int(lava_count * self.goal_lava_ratio))

        # Find all PATH tiles in interior (candidates for goal placement)
        path_tiles = []
        for iy in range(1, size - 1):
            for ix in range(1, size - 1):
                if grid[iy][ix] == tc.PATH:
                    path_tiles.append((ix, iy))

        # Place goals on random PATH tiles
        random.shuffle(path_tiles)
        goals = []
        for i in range(min(num_goals, len(path_tiles))):
            gx, gy = path_tiles[i]
            grid[gy][gx] = tc.GOAL
            goals.append((gx, gy))

        # Compute start position (center-nearest PATH tile)
        start = Maze._find_center_start(grid, tc)

        return Maze(grid=grid, start=start, goals=goals, tile_config=tc)
