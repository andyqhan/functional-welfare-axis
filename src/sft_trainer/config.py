"""Configuration for SFT maze training."""

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.pytorch_trainer.config import TILE_PRESETS


@dataclass
class SFTConfig:
    # Model
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    use_lora: bool = True  # False = full fine-tune, ignores lora_rank/lora_alpha/target_modules
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: str = "all-linear"
    # Override tokenizer.chat_template with the one from this model (e.g.
    # "Qwen/Qwen3-4B-Instruct-2507" when training a Base model, to avoid the
    # Base template's inconsistent thinking-block wrapping across turns).
    chat_template_from: str = ""

    # Dataset
    num_train: int = 50_000
    num_val: int = 1_000
    maze_size: int = 31
    max_turns: int = 15
    goal_lava_ratio: float = 0.20
    trajectory_algorithm: str = "perfect"
    dataset_dir: str = "datasets/sft"
    dataset_workers: int = 8
    max_reachable_goals: int = 15
    base_seed: int = 42

    # Environment
    shuffle_directions: bool = True
    relative_directions: bool = True
    tile_preset: str = "office"

    # Rewards
    step_penalty: float = -0.1
    goal_reward: float = 20.0
    lava_penalty: float = -10.0

    # Training
    lr: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True

    # Evaluation
    eval_mazes: int = 200
    eval_max_turns: int = 15

    # Checkpoints
    save_steps: int = 500
    eval_steps: int = 500

    # Infra
    output_dir: str = "runs/sft"
    wandb_project: str = "lava-maze-rl"
    wandb_entity: str = ""

    @property
    def tile_chars(self) -> dict:
        return TILE_PRESETS[self.tile_preset]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SFTConfig":
        data = json.loads(Path(path).read_text())
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="SFT Maze Training")
    for f in fields(SFTConfig):
        if f.type == bool:
            parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("true", "1", "yes"),
                                default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    args = parser.parse_args()
    return SFTConfig(**vars(args))
