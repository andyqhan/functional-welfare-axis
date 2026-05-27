"""Tile configuration for maze rendering.

Supports multiple tile representation modes:
- emoji: PATH="🍩", LAVA="🧁", GOAL="🍨", PLAYER="😀"
- letters: PATH="X", LAVA="Y", GOAL="Z", PLAYER="@"
"""

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TileConfig:
    """Immutable configuration for maze tile representations."""

    PATH: str
    LAVA: str
    GOAL: str
    PLAYER: str
    mode: str  # "emoji" or "letters"

    @classmethod
    def emoji(cls) -> "TileConfig":
        """Create emoji mode tile configuration."""
        return cls(PATH="🍩", LAVA="🧁", GOAL="🍨", PLAYER="😀", mode="emoji")

    @classmethod
    def letters(cls) -> "TileConfig":
        """Create letters mode tile configuration."""
        return cls(PATH="X", LAVA="Y", GOAL="Z", PLAYER="@", mode="letters")

    @classmethod
    def from_mode(cls, mode: str) -> "TileConfig":
        """Create tile configuration from mode name.

        Args:
            mode: Either "emoji" or "letters"

        Returns:
            TileConfig for the specified mode

        Raises:
            ValueError: If mode is not recognized
        """
        if mode == "emoji":
            return cls.emoji()
        elif mode == "letters":
            return cls.letters()
        else:
            raise ValueError(f"Unknown tile mode: {mode!r}. Must be 'emoji' or 'letters'.")

    def get_english_name(self, tile_type: str) -> str:
        """Get English name for a tile character via unicodedata.

        Args:
            tile_type: One of "LAVA", "PATH", or "GOAL"

        Returns:
            Lowercased Unicode name (e.g. "cupcake") or the character itself for ASCII.
        """
        char = getattr(self, tile_type)
        if len(char) == 1 and char.isascii():
            return char
        try:
            return unicodedata.name(char).lower()
        except (ValueError, TypeError):
            return char

    @classmethod
    def from_custom_config(cls, custom_config: dict) -> "TileConfig":
        """Create TileConfig from YAML custom config section.

        Supports tile_mode with optional tile_chars overrides. If tile_chars is
        present, those characters override the defaults for the given mode.

        Args:
            custom_config: Dict with 'tile_mode' (required) and optional 'tile_chars'
                          mapping of {path, lava, goal, player} to custom characters.

        Returns:
            TileConfig with the specified or default characters.
        """
        mode = custom_config.get("tile_mode", "emoji")
        base = cls.from_mode(mode)

        tile_chars = custom_config.get("tile_chars")
        if not tile_chars:
            return base

        return cls(
            PATH=tile_chars.get("path", base.PATH),
            LAVA=tile_chars.get("lava", base.LAVA),
            GOAL=tile_chars.get("goal", base.GOAL),
            PLAYER=tile_chars.get("player", base.PLAYER),
            mode=mode,
        )


# Default tile config (emoji mode) for backward compatibility
DEFAULT_TILE_CONFIG = TileConfig.emoji()
