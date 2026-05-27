# Maze environment package
from .maze import Maze, MazeGenerator, DEFAULT_SIZE, PATH, LAVA, GOAL
from .game import Game, GameResult

__all__ = [
    "Maze",
    "MazeGenerator",
    "DEFAULT_SIZE",
    "PATH",
    "LAVA",
    "GOAL",
    "Game",
    "GameResult",
]
