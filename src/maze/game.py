import random
from dataclasses import dataclass, field
from enum import Enum

# Support both direct execution and module execution
try:
    from .maze import Maze
except ImportError:
    from maze import Maze

DEFAULT_PROMPT_TEMPLATE = (
    "You are in a maze. Currently, you're at location ({x}, {y}). "
    "{directions_text}Where do you move? Output only N/E/S/W without punctuation."
)


class GameResult(Enum):
    CONTINUE = "continue"
    WIN = "win"
    LOSE = "lose"


# Direction deltas: name -> (dx, dy)
DIRECTIONS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}

OPPOSITE_DIRECTIONS = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

RELATIVE_PHRASES = {
    "north": "above",
    "east": "to the right",
    "south": "below",
    "west": "to the left",
}


@dataclass
class Game:
    maze: Maze
    position: tuple[int, int] | None = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    wind_frequency: float = 0.0  # Probability of wind (0 = off, 0.1 = 10%)
    melting_path: bool = False  # When True, tiles melt into lava after the agent steps off
    direction_order: list[str] | None = field(default=None, repr=False)
    relative_directions: bool = False
    _terminal: bool = field(default=False, repr=False)
    _result: GameResult = field(default=GameResult.CONTINUE, repr=False)
    _lava_visit_count: int = field(default=0, repr=False)
    _goal_visit_count: int = field(default=0, repr=False)
    _consumed_goals: set = field(default_factory=set, repr=False)  # Consumed goal positions
    _melted_tiles: set = field(default_factory=set, repr=False)  # Tiles that melted into lava
    _pending_wind_message: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.position is None:
            self.position = self.maze.start
        if self.direction_order is None:
            self.direction_order = ["north", "east", "south", "west"]

    def get_tile(self, x: int, y: int) -> str:
        """Get tile at position, accounting for consumed goals and melted tiles."""
        if (x, y) in self._consumed_goals:
            if (x, y) in self._melted_tiles:
                return self.maze.tile_config.LAVA
            return self.maze.tile_config.PATH
        if (x, y) in self._melted_tiles:
            return self.maze.tile_config.LAVA
        return self.maze.get_tile(x, y)

    def get_available_directions(self) -> dict[str, str]:
        """Returns dict of direction name -> tile emoji for valid directions."""
        x, y = self.position
        available = {}

        for name, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if self.maze.is_in_bounds(nx, ny):
                available[name] = self.get_tile(nx, ny)

        return available

    def get_prompt(self) -> str:
        """Generate the prompt for the current game state."""
        available = self.get_available_directions()

        # Build directions text
        parts = []
        for direction in self.direction_order:
            if direction in available:
                if self.relative_directions:
                    phrase = RELATIVE_PHRASES[direction]
                else:
                    phrase = f"to the {direction}"
                parts.append(f"{phrase} there is {available[direction]}")

        if parts:
            # Capitalize first letter
            parts[0] = parts[0][0].upper() + parts[0][1:]
            directions_text = "; ".join(parts) + ". "
        else:
            directions_text = ""

        # Prepend wind message if present
        base_prompt = self.prompt_template.format(
            x=self.position[0], y=self.position[1], directions_text=directions_text
        )
        if self._pending_wind_message:
            return f"{self._pending_wind_message} {base_prompt}"
        return base_prompt

    def _apply_wind(self) -> GameResult | None:
        """Maybe apply wind after a move. Returns GameResult if wind changes state."""
        if self.wind_frequency <= 0 or random.random() >= self.wind_frequency:
            return None

        # Pick random wind direction (where wind comes FROM)
        wind_from = random.choice(list(DIRECTIONS.keys()))
        push_direction = OPPOSITE_DIRECTIONS[wind_from]

        dx, dy = DIRECTIONS[push_direction]
        nx, ny = self.position[0] + dx, self.position[1] + dy

        # If push is blocked by boundary, no wind effect
        if not self.maze.is_in_bounds(nx, ny):
            return None

        # Melt the tile we're leaving (if enabled)
        if self.melting_path:
            old_tile = self.get_tile(*self.position)
            if old_tile == self.maze.tile_config.PATH:
                self._melted_tiles.add(self.position)

        # Apply push
        self.position = (nx, ny)
        self._pending_wind_message = (
            f"A strong wind from the {wind_from.capitalize()} blew you {push_direction.capitalize()}!"
        )

        # Check tile and update state (use get_tile to account for consumed goals)
        tile = self.get_tile(nx, ny)
        tc = self.maze.tile_config
        if tile == tc.LAVA:
            self._lava_visit_count += 1
            return GameResult.LOSE
        elif tile == tc.GOAL:
            # Consume the goal
            self._consumed_goals.add((nx, ny))
            self._goal_visit_count += 1
            return GameResult.WIN
        return GameResult.CONTINUE

    def move(self, direction: str) -> GameResult:
        """Execute a move. Returns the result of the move."""
        direction = direction.lower().strip()

        # Handle short forms
        short_map = {"n": "north", "s": "south", "e": "east", "w": "west"}
        if direction in short_map:
            direction = short_map[direction]

        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")

        dx, dy = DIRECTIONS[direction]
        nx, ny = self.position[0] + dx, self.position[1] + dy

        if not self.maze.is_in_bounds(nx, ny):
            raise ValueError(f"Cannot move {direction}: out of bounds")

        # Clear previous wind message
        self._pending_wind_message = None

        # Melt the tile we're leaving (if enabled)
        if self.melting_path:
            old_tile = self.get_tile(*self.position)
            if old_tile == self.maze.tile_config.PATH:
                self._melted_tiles.add(self.position)

        self.position = (nx, ny)
        # Use get_tile to account for consumed goals
        tile = self.get_tile(nx, ny)
        tc = self.maze.tile_config

        if tile == tc.LAVA:
            self._lava_visit_count += 1
            result = GameResult.LOSE
        elif tile == tc.GOAL:
            # Consume the goal
            self._consumed_goals.add((nx, ny))
            self._goal_visit_count += 1
            result = GameResult.WIN
        else:
            result = GameResult.CONTINUE

        # Apply wind effect (may change position and override result)
        wind_result = self._apply_wind()
        if wind_result is not None:
            return wind_result

        return result

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self._terminal

    def get_result(self) -> GameResult:
        """Get the final result of the game."""
        return self._result

    def get_lava_visit_count(self) -> int:
        """Get the number of times lava tiles were visited."""
        return self._lava_visit_count

    def get_goal_visit_count(self) -> int:
        """Get the number of goals that have been collected."""
        return self._goal_visit_count

    def get_total_goal_count(self) -> int:
        """Get the total number of goals in the maze."""
        return len(self.maze.goals)

    def get_consumed_goals(self) -> set[tuple[int, int]]:
        """Get the set of consumed goal positions."""
        return self._consumed_goals.copy()

    def get_melted_tiles(self) -> set[tuple[int, int]]:
        """Get the set of tiles that have melted into lava."""
        return self._melted_tiles.copy()

    def render(self, show_player: bool = True) -> str:
        """Render the maze with optional player position marker."""
        lines = []
        for y, row in enumerate(self.maze.grid):
            row_chars = []
            for x, tile in enumerate(row):
                if show_player and (x, y) == self.position:
                    row_chars.append(self.maze.tile_config.PLAYER)
                else:
                    # Use get_tile to show consumed goals as path
                    row_chars.append(self.get_tile(x, y))
            lines.append("".join(row_chars))
        return "\n".join(lines)
