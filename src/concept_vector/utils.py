"""Shared utilities for concept vector extraction and exploration."""

import json
from datetime import datetime
from pathlib import Path

from src.maze.tiles import TileConfig
from src.pytorch_trainer.config import TILE_PRESETS


def generate_output_suffix(run_id: str | None) -> str:
    """Generate output filename suffix: <run_id>_<timestamp> or just <timestamp>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_id:
        return f"{run_id}_{timestamp}"
    return timestamp


def load_tile_config_from_run_config(checkpoint_path: Path | str) -> TileConfig:
    """Load TileConfig from a training run's config.json.

    Walks up from checkpoint_path looking for a config.json that matches a
    known trainer format (PyTorch / Tinker / SFT / veRL). Non-matching
    config.json files (e.g. the HF model config sitting next to checkpoint
    weights) are skipped — the search continues up the directory tree.

    Args:
        checkpoint_path: Path to checkpoint (e.g., runs/maze-grpo-xxx/global_step_N/actor)

    Returns:
        TileConfig from the training config. Falls back to `TileConfig.emoji()`
        with a stderr warning if no trainer config is found within 6 levels.
    """
    checkpoint_path = Path(checkpoint_path)

    current = checkpoint_path
    for _ in range(6):  # Search up to 6 levels
        config_path = current / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            # PyTorch trainer / Tinker format: flat keys with tile_mode/tile_chars at top level
            if ("model_path" in config or "model_name" in config) and config.get("tile_mode"):
                return TileConfig.from_custom_config(config)

            # SFT trainer format: tile_preset key (e.g. "office", "desserts")
            if ("model_path" in config or "model_name" in config) and config.get("tile_preset"):
                preset = TILE_PRESETS[config["tile_preset"]]
                return TileConfig(
                    PATH=preset["path"], LAVA=preset["lava"],
                    GOAL=preset["goal"], PLAYER=preset.get("player", "😀"),
                    mode="emoji",
                )

            # veRL format: nested under actor_rollout_ref.rollout.custom
            custom = (
                config.get("actor_rollout_ref", {})
                .get("rollout", {})
                .get("custom", {})
            )
            if custom.get("tile_mode"):
                return TileConfig.from_custom_config(custom)
            # No trainer-format match (probably the HF model config); keep walking up.
        if current == current.parent:  # Reached filesystem root
            break
        current = current.parent

    import sys
    print(
        f"[load_tile_config_from_run_config] WARN: no trainer config.json found "
        f"walking up from {checkpoint_path}; falling back to TileConfig.emoji() "
        f"(PATH='🍩', LAVA='🧁', GOAL='🍨'). If this checkpoint was trained on a "
        f"different preset, downstream extraction will silently use the wrong tiles.",
        file=sys.stderr,
    )
    return TileConfig.emoji()


def load_tile_config_from_cv_dir(cv_dir: Path) -> dict | None:
    """Load tile_mode and tile_chars from the training run's config.json.

    The concept vector directory contains metadata.json which references
    the checkpoint_dir. We look for config.json in the parent run directory.
    Supports both veRL (nested) and pytorch trainer (flat) config formats,
    with fallback to metadata.json tile_config.

    Args:
        cv_dir: Path to concept vector directory containing metadata.json

    Returns:
        Dict with "tile_mode" and optional "tile_chars", or None if not found
    """
    # Load concept vector metadata to get checkpoint path
    metadata_path = cv_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    checkpoint_path = metadata.get("checkpoint_path")
    if checkpoint_path:
        # checkpoint_path is like runs/maze-grpo-123/global_step_5/actor/lora_adapter
        # The config.json is in the run directory: runs/maze-grpo-123/config.json
        checkpoint_dir = Path(checkpoint_path)

        # Navigate up to find the run directory (which contains config.json)
        current = checkpoint_dir
        for _ in range(6):  # Search up to 6 levels
            config_path = current / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                # PyTorch trainer / Tinker format: flat keys at top level
                if ("model_path" in config or "model_name" in config) and config.get("tile_mode"):
                    result = {"tile_mode": config["tile_mode"]}
                    if config.get("tile_chars"):
                        result["tile_chars"] = config["tile_chars"]
                    return result

                # SFT trainer format: tile_preset key
                if ("model_path" in config or "model_name" in config) and config.get("tile_preset"):
                    preset = TILE_PRESETS[config["tile_preset"]]
                    return {
                        "tile_mode": "emoji",
                        "tile_chars": {k: v for k, v in preset.items() if k != "player"},
                    }

                # veRL format: nested under actor_rollout_ref.rollout.custom
                custom = (
                    config.get("actor_rollout_ref", {})
                    .get("rollout", {})
                    .get("custom", {})
                )
                tile_mode = custom.get("tile_mode")
                if tile_mode:
                    result = {"tile_mode": tile_mode}
                    tile_chars = custom.get("tile_chars")
                    if tile_chars:
                        result["tile_chars"] = tile_chars
                    return result
            current = current.parent
            if current == current.parent:  # Reached root
                break

    # Fallback: read tile_config from metadata.json itself (saved during extraction)
    tile_config = metadata.get("tile_config")
    if tile_config and tile_config.get("mode"):
        result = {"tile_mode": tile_config["mode"]}
        if tile_config.get("tile_chars"):
            result["tile_chars"] = tile_config["tile_chars"]
        return result

    return None
