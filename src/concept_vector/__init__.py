"""
Concept vector extraction for maze trajectories.

This module extracts activation differences between trajectories ending on
LAVA tiles vs. other tiles (PATH, GOAL), creating a "concept vector" that
represents the model's internal representation of lava-ending trajectories.
"""

from .activation_extraction import get_activations, get_mean_activations, get_mean_diff, get_all_tile_activations
from .model_utils import load_model_with_lora, format_trajectory

__all__ = [
    "get_activations",
    "get_mean_activations",
    "get_mean_diff",
    "get_all_tile_activations",
    "load_model_with_lora",
    "format_trajectory",
]
