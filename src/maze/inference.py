"""Standalone maze inference for evaluating LoRA checkpoints on fresh mazes.

Runs batched multi-turn maze games using HuggingFace generate() without
requiring the full veRL training infrastructure.

Usage:
    uv run python -m src.maze.inference --checkpoint runs/<run>/global_step_N --num-mazes 20
    uv run python -m src.maze.inference --checkpoint runs/<run>/global_step_N/actor --base-model-only
"""

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import LogitsProcessor

from src.concept_vector.model_utils import load_model_with_lora, load_base_model
from src.concept_vector.utils import load_tile_config_from_run_config
from src.maze.game import DIRECTIONS, Game, GameResult
from src.maze.maze import MazeGenerator


class DirectionLogitsProcessor(LogitsProcessor):
    """Masks logits to only allow N/E/S/W tokens."""

    def __init__(self, direction_token_ids: list[int]):
        assert len(direction_token_ids) == 4, (
            f"Expected 4 direction token IDs, got {len(direction_token_ids)}"
        )
        self.direction_token_ids = direction_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.direction_token_ids:
            mask[:, tid] = 0.0
        return scores + mask


# ─── Checkpoint / config resolution ─────────────────────────────────────────

def resolve_checkpoint_path(path: str) -> Path:
    """Normalise user-supplied checkpoint path to the directory containing lora_adapter/.

    Accepted inputs (veRL format):
        global_step_N/           -> appends actor/
        global_step_N/actor/     -> used as-is
        global_step_N/actor/lora_adapter/  -> uses parent (actor/)

    Accepted inputs (PyTorch trainer format):
        global_step_N/           -> used as-is (lora_adapter/ directly inside)
        global_step_N/lora_adapter/  -> uses parent

    Returns the directory containing lora_adapter/.
    """
    p = Path(path)
    assert p.exists(), f"Path does not exist: {p}"

    if p.name == "lora_adapter":
        p = p.parent
    elif p.name != "actor":
        # Try veRL format: global_step_N/actor/
        candidate = p / "actor"
        if candidate.exists() and (candidate / "lora_adapter").exists():
            p = candidate
        # Otherwise assume pytorch format: lora_adapter/ directly in p

    assert (p / "lora_adapter").exists(), f"lora_adapter/ not found in {p}"
    return p


def load_run_config(checkpoint_path: Path) -> dict:
    """Walk up from *checkpoint_path* to find config.json and extract relevant fields.

    Returns a dict with keys: base_model, rewards, maze_size, goal_lava_ratio.
    Falls back to sensible defaults when config.json is absent.
    """
    defaults = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "rewards": {"step_penalty": -0.1, "goal_reward": 10.0, "lava_penalty": -10.0},
        "maze_size": 100,
        "goal_lava_ratio": 0.20,
    }

    current = checkpoint_path
    for _ in range(6):
        config_path = current / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

            # Detect format: pytorch trainer has flat "model_path" key
            if "model_path" in cfg:
                # PyTorch trainer format (flat keys)
                return {
                    "base_model": cfg.get("model_path", defaults["base_model"]),
                    "rewards": {
                        "step_penalty": cfg.get("step_penalty", -0.1),
                        "goal_reward": cfg.get("goal_reward", 10.0),
                        "lava_penalty": cfg.get("lava_penalty", -10.0),
                    },
                    "maze_size": cfg.get("maze_size", defaults["maze_size"]),
                    "goal_lava_ratio": cfg.get("goal_lava_ratio", defaults["goal_lava_ratio"]),
                    "action_masking": cfg.get("action_masking", False),
                    "relative_directions": cfg.get("relative_directions", False),
                    "melting_path": cfg.get("melting_path", False),
                    "enable_thinking": cfg.get("enable_thinking", None),
                }

            # Tinker trainer format (flat keys, uses "model_name" + "renderer_name")
            if "model_name" in cfg and "renderer_name" in cfg:
                return {
                    "base_model": cfg.get("model_name", defaults["base_model"]),
                    "rewards": {
                        "step_penalty": cfg.get("step_penalty", -0.1),
                        "goal_reward": cfg.get("goal_reward", 10.0),
                        "lava_penalty": cfg.get("lava_penalty", -10.0),
                    },
                    "maze_size": cfg.get("maze_size", defaults["maze_size"]),
                    "goal_lava_ratio": cfg.get("goal_lava_ratio", defaults["goal_lava_ratio"]),
                    "action_masking": False,
                    "relative_directions": cfg.get("relative_directions", False),
                    "melting_path": cfg.get("melting_path", False),
                    "enable_thinking": cfg.get("enable_thinking", None),
                }

            # veRL format (nested keys)
            actor = cfg.get("actor_rollout_ref", {})
            rollout = actor.get("rollout", {})
            custom = rollout.get("custom", {}) or {}
            data = cfg.get("data", {})
            maze_cfg = data.get("maze", {}) or {}

            rewards_cfg = custom.get("rewards", {}) or {}
            return {
                "base_model": actor.get("model", {}).get("path", defaults["base_model"]),
                "rewards": {
                    "step_penalty": rewards_cfg.get("step_penalty", -0.1),
                    "goal_reward": rewards_cfg.get("goal_reward", 10.0),
                    "lava_penalty": rewards_cfg.get("lava_penalty", -10.0),
                },
                "maze_size": maze_cfg.get("size", defaults["maze_size"]),
                "goal_lava_ratio": maze_cfg.get("goal_lava_ratio", defaults["goal_lava_ratio"]),
            }
        current = current.parent
        if current == current.parent:
            break

    print("WARNING: config.json not found, using defaults")
    return defaults


# ─── Direction parsing ───────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DIRECTIONS = {"N", "E", "S", "W"}
_DIRECTION_WORDS = {"North": "N", "South": "S", "East": "E", "West": "W"}


def parse_direction(text: str) -> str | None:
    """Extract a direction (N/E/S/W) from model output.

    Matches the training parser in src/pytorch_trainer/rollout.py:
    1. Strip <think>...</think> blocks (Qwen3 thinking tokens).
    2. Exact match in {N, E, S, W}.
    3. Exact match in {North, South, East, West}.
    4. First char is a valid direction AND second char (if any) is not lowercase
       (avoids "No", "Sure", etc. while accepting "N.", "NNN", "N\\n").
    5. Return None if unparseable.
    """
    # Strip thinking blocks
    cleaned = _THINK_RE.sub("", text)
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # Exact single-letter direction
    if cleaned in _DIRECTIONS:
        return cleaned

    # Exact full-word direction
    if cleaned in _DIRECTION_WORDS:
        return _DIRECTION_WORDS[cleaned]

    # First-char fallback: accept if first char is a direction and
    # the next char (if any) is not lowercase (matches training parser)
    if (
        cleaned[0] in _DIRECTIONS
        and (len(cleaned) < 2 or not cleaned[1].islower())
    ):
        return cleaned[0]

    return None


# ─── Reward computation ──────────────────────────────────────────────────────

def compute_reward(trajectory_len: int, goal_count: int, lava_count: int,
                   rewards_cfg: dict) -> float:
    """Same formula as src/verl/reward_async.py."""
    return (
        rewards_cfg["step_penalty"] * trajectory_len
        + rewards_cfg["goal_reward"] * goal_count
        + rewards_cfg["lava_penalty"] * lava_count
    )


# ─── Batched inference loop ─────────────────────────────────────────────────

def run_batched_inference(
    model,
    tokenizer,
    games: list[Game],
    mazes: list,
    seeds: list[int],
    *,
    max_turns: int = 15,
    batch_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    rewards_cfg: dict,
    action_masking: bool = False,
    enable_thinking: bool | None = None,
) -> list[dict]:
    """Run multi-turn maze games in batches.

    Returns a list of result dicts, one per game.
    """
    n = len(games)
    assert n == len(mazes) == len(seeds)

    # Set up action masking if enabled
    processor = None
    id_to_letter = {}
    if action_masking:
        direction_tokens = {}
        for letter in ["N", "E", "S", "W"]:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"Direction '{letter}' encodes to {len(ids)} tokens: {ids}"
            direction_tokens[letter] = ids[0]
        processor = DirectionLogitsProcessor([direction_tokens[d] for d in ["N", "E", "S", "W"]])
        id_to_letter = {v: k for k, v in direction_tokens.items()}
        print(f"Action masking enabled: N={direction_tokens['N']}, E={direction_tokens['E']}, S={direction_tokens['S']}, W={direction_tokens['W']}")

    short_map = {"N": "north", "S": "south", "E": "east", "W": "west"}

    # Per-game state
    conversations = [[] for _ in range(n)]  # list of messages per game
    trajectories = [[game.position] for game in games]
    active = list(range(n))  # indices of still-running games
    termination_reasons = ["max_turns"] * n
    num_turns = [0] * n
    invalid_counts = [0] * n
    last_invalid_responses = [None] * n  # raw text of last unparseable response

    # Initialise with first prompt
    for i in range(n):
        prompt = games[i].get_prompt()
        conversations[i].append({"role": "user", "content": prompt})

    pad_token_id = tokenizer.pad_token_id

    for turn in range(max_turns):
        if not active:
            break

        # Process active games in sub-batches
        for batch_start in range(0, len(active), batch_size):
            batch_indices = active[batch_start : batch_start + batch_size]

            # Build token inputs via chat template
            batch_input_ids = []
            template_kwargs = {}
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            for idx in batch_indices:
                input_ids = tokenizer.apply_chat_template(
                    conversations[idx],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    **template_kwargs,
                ).squeeze(0)
                batch_input_ids.append(input_ids)

            # Left-pad to uniform length
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_ids = []
            attention_masks = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
                    padded = torch.cat([padding, ids])
                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(len(ids), dtype=torch.long),
                    ])
                else:
                    padded = ids
                    mask = torch.ones(len(ids), dtype=torch.long)
                padded_ids.append(padded)
                attention_masks.append(mask)

            batched_input_ids = torch.stack(padded_ids).to(model.device)
            batched_attention_mask = torch.stack(attention_masks).to(model.device)

            if action_masking:
                with torch.no_grad():
                    output_ids = model.generate(
                        batched_input_ids,
                        attention_mask=batched_attention_mask,
                        max_new_tokens=1,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=pad_token_id,
                        logits_processor=[processor],
                    )

                # Process each response — guaranteed to be a valid direction
                for j, idx in enumerate(batch_indices):
                    token_id = output_ids[j, -1].item()
                    assert token_id in id_to_letter, (
                        f"Generated token {token_id} is not a direction token. "
                        f"Expected one of {list(id_to_letter.keys())}"
                    )
                    direction = id_to_letter[token_id]
                    dir_name = short_map[direction]

                    # Validate move is in-bounds (wall bounce — still a valid turn)
                    dx, dy = DIRECTIONS[dir_name]
                    nx, ny = games[idx].position[0] + dx, games[idx].position[1] + dy
                    if not games[idx].maze.is_in_bounds(nx, ny):
                        invalid_counts[idx] += 1
                        termination_reasons[idx] = "invalid_output"
                        conversations[idx].append({"role": "assistant", "content": direction})
                        continue

                    games[idx].move(dir_name)
                    num_turns[idx] += 1
                    trajectories[idx].append(games[idx].position)
                    conversations[idx].append({"role": "assistant", "content": direction})

                    if turn < max_turns - 1:
                        next_prompt = games[idx].get_prompt()
                        conversations[idx].append({"role": "user", "content": next_prompt})
            else:
                with torch.no_grad():
                    output_ids = model.generate(
                        batched_input_ids,
                        attention_mask=batched_attention_mask,
                        max_new_tokens=256,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=pad_token_id,
                    )

                # Process each response
                for j, idx in enumerate(batch_indices):
                    response_ids = output_ids[j][max_len:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                    direction = parse_direction(response_text)
                    if direction is None:
                        # Unparseable -> end game
                        invalid_counts[idx] += 1
                        termination_reasons[idx] = "invalid_output"
                        last_invalid_responses[idx] = response_text.strip()
                        conversations[idx].append({"role": "assistant", "content": response_text.strip()})
                        continue

                    # Validate move is in-bounds
                    dir_name = short_map[direction]
                    dx, dy = DIRECTIONS[dir_name]
                    nx, ny = games[idx].position[0] + dx, games[idx].position[1] + dy
                    if not games[idx].maze.is_in_bounds(nx, ny):
                        invalid_counts[idx] += 1
                        termination_reasons[idx] = "invalid_output"
                        last_invalid_responses[idx] = response_text.strip()
                        conversations[idx].append({"role": "assistant", "content": response_text.strip()})
                        continue

                    games[idx].move(dir_name)
                    num_turns[idx] += 1
                    trajectories[idx].append(games[idx].position)
                    conversations[idx].append({"role": "assistant", "content": response_text.strip()})

                    if turn < max_turns - 1:
                        next_prompt = games[idx].get_prompt()
                        conversations[idx].append({"role": "user", "content": next_prompt})

        # Remove finished games from active set (invalid_output terminates)
        active = [
            i for i in active
            if termination_reasons[i] == "max_turns"
        ]

    # Build results
    results = []
    for i in range(n):
        traj_len = len(trajectories[i])
        goal_count = games[i].get_goal_visit_count()
        lava_count = games[i].get_lava_visit_count()
        total_goals = games[i].get_total_goal_count()
        optimal = mazes[i].get_optimal_path_length()

        reward = compute_reward(traj_len, goal_count, lava_count, rewards_cfg)

        results.append({
            "maze_id": i,
            "seed": seeds[i],
            "final_game_result": "continue" if termination_reasons[i] == "max_turns" else "invalid",
            "player_trajectory": trajectories[i],
            "trajectory_length": traj_len,
            "lava_visit_count": lava_count,
            "goal_visit_count": goal_count,
            "total_goals": total_goals,
            "optimal_path_length": optimal,
            "reward": reward,
            "num_turns": num_turns[i],
            "invalid_output_count": invalid_counts[i],
            "termination_reason": termination_reasons[i],
            "last_invalid_response": last_invalid_responses[i],
            "messages": conversations[i],
            "maze_serialization": mazes[i].serialize(),
        })

    return results


# ─── Summary statistics ──────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print aggregate statistics to stdout."""
    n = len(results)
    rewards = [r["reward"] for r in results]
    turns = [r["num_turns"] for r in results]
    goals = [r["goal_visit_count"] for r in results]
    total_goals = [r["total_goals"] for r in results]
    lava_hits = [r["lava_visit_count"] for r in results]
    traj_lens = [r["trajectory_length"] for r in results]
    invalid_counts = [r["invalid_output_count"] for r in results]

    goal_pcts = [g / t * 100 if t > 0 else 0 for g, t in zip(goals, total_goals)]

    # Efficiency: trajectory length / optimal path length (lower is better)
    efficiencies = []
    for r in results:
        opt = r["optimal_path_length"]
        if opt and opt > 0:
            efficiencies.append(r["trajectory_length"] / opt)

    term_counts = {}
    for r in results:
        reason = r["termination_reason"]
        term_counts[reason] = term_counts.get(reason, 0) + 1

    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Games:              {n}")
    for reason, count in sorted(term_counts.items()):
        print(f"    {reason}: {count}")
    print()
    print(f"  Reward:             mean={statistics.mean(rewards):.2f}  median={statistics.median(rewards):.2f}  min={min(rewards):.2f}  max={max(rewards):.2f}")
    print(f"  Goals collected:    mean={statistics.mean(goals):.1f}/{statistics.mean(total_goals):.0f}  ({statistics.mean(goal_pcts):.1f}%)")
    print(f"  Lava hits:          mean={statistics.mean(lava_hits):.2f}  total={sum(lava_hits)}")
    print(f"  Invalid outputs:    total={sum(invalid_counts)}")
    print(f"  Turns used:         mean={statistics.mean(turns):.1f}  max={max(turns)}")
    print(f"  Trajectory length:  mean={statistics.mean(traj_lens):.1f}")
    if efficiencies:
        print(f"  Efficiency ratio:   mean={statistics.mean(efficiencies):.2f}  (traj_len / optimal, lower=better)")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Maze inference for LoRA checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--output", default=None, help="Output directory (auto-derived if omitted)")
    parser.add_argument("--num-mazes", type=int, default=20, help="Number of mazes to evaluate")
    parser.add_argument("--max-turns", type=int, default=15, help="Max turns per game")
    parser.add_argument("--batch-size", type=int, default=8, help="Parallel inference batch size")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for maze generation")
    parser.add_argument("--wind-frequency", type=float, default=0.0, help="Wind effect frequency")
    parser.add_argument("--base-model-only", action="store_true", help="Run base model without LoRA")
    parser.add_argument("--action-masking", action="store_true", default=None, help="Constrain sampling to N/E/S/W tokens (auto-detected from config if omitted)")
    parser.add_argument("--no-action-masking", action="store_true", help="Disable action masking even if config has it enabled")
    parser.add_argument("--relative-directions", action="store_true", default=None, help="Use relative directions (auto-detected from config if omitted)")
    parser.add_argument("--no-relative-directions", action="store_true", help="Disable relative directions even if config has it enabled")
    parser.add_argument("--melting-path", action="store_true", default=None, help="Enable melting path (auto-detected from config if omitted)")
    parser.add_argument("--no-melting-path", action="store_true", help="Disable melting path even if config has it enabled")
    parser.add_argument("--enable-thinking", action="store_true", default=None, help="Enable thinking mode (auto-detected from config if omitted)")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode (inject empty think block)")
    parser.add_argument("--maze-size", type=int, default=None, help="Override maze size from config")
    parser.add_argument("--goal-lava-ratio", type=float, default=None, help="Override goal-lava ratio")
    args = parser.parse_args()

    t_start = time.time()

    # ── Resolve checkpoint and config ──
    if args.base_model_only:
        checkpoint_path = Path(args.checkpoint)
        # For base-model-only, checkpoint is just used for config lookup
        run_config = load_run_config(checkpoint_path)
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        run_config = load_run_config(checkpoint_path)

    rewards_cfg = run_config["rewards"]
    maze_size = args.maze_size or run_config["maze_size"]
    goal_lava_ratio = args.goal_lava_ratio if args.goal_lava_ratio is not None else run_config["goal_lava_ratio"]
    base_model = run_config["base_model"]

    # Resolve action masking: explicit CLI flag > auto-detect from config
    if args.no_action_masking:
        action_masking = False
    elif args.action_masking is not None:
        action_masking = args.action_masking
    else:
        action_masking = run_config.get("action_masking", False)
        if action_masking:
            print(f"Auto-detected action_masking=true from run config")

    # Resolve relative directions: explicit CLI flag > auto-detect from config
    if args.no_relative_directions:
        relative_directions = False
    elif args.relative_directions is not None:
        relative_directions = args.relative_directions
    else:
        relative_directions = run_config.get("relative_directions", False)
        if relative_directions:
            print(f"Auto-detected relative_directions=true from run config")

    # Resolve melting path: explicit CLI flag > auto-detect from config
    if args.no_melting_path:
        melting_path = False
    elif args.melting_path is not None:
        melting_path = args.melting_path
    else:
        melting_path = run_config.get("melting_path", False)
        if melting_path:
            print(f"Auto-detected melting_path=true from run config")

    # Resolve enable_thinking: explicit CLI flag > auto-detect from config
    # None means "not set" (no thinking model), False means "no-thinking mode" (inject empty think block)
    if args.no_thinking:
        enable_thinking = False
    elif args.enable_thinking is not None:
        enable_thinking = args.enable_thinking or None  # True -> None (default behavior), False stays False
    else:
        enable_thinking = run_config.get("enable_thinking", None)
        if enable_thinking is False:
            print(f"Auto-detected enable_thinking=false from run config (no-thinking mode)")

    # ── Determine tile config ──
    tile_config = load_tile_config_from_run_config(checkpoint_path)

    # ── Output directory ──
    if args.output:
        output_dir = Path(args.output)
    else:
        # Auto-derive: artifacts/inference/<run-name>_<step>/
        # Walk up to find run dir name and step
        parts = []
        p = checkpoint_path
        for _ in range(6):
            if p.name.startswith("global_step_"):
                parts.insert(0, p.name.replace("global_step_", "step"))
            elif p.name.startswith("runs"):
                break
            elif p.name not in ("actor", "lora_adapter", "huggingface"):
                parts.insert(0, p.name)
            p = p.parent
        dirname = "_".join(parts) if parts else "inference"
        if args.base_model_only:
            dirname += "_base"
        output_dir = Path("artifacts/inference") / dirname

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Load model ──
    print(f"\n{'=' * 60}")
    if args.base_model_only:
        print(f"Loading base model (no LoRA): {base_model}")
        model, tokenizer = load_base_model(base_model)
    else:
        print(f"Loading model with LoRA from: {checkpoint_path}")
        model, tokenizer = load_model_with_lora(str(checkpoint_path), base_model)
    print(f"{'=' * 60}\n")

    # ── Generate mazes ──
    print(f"Generating {args.num_mazes} mazes (size={maze_size}, goal_lava_ratio={goal_lava_ratio}, tile_mode={tile_config.mode})")
    generator = MazeGenerator(size=maze_size, goal_lava_ratio=goal_lava_ratio, tile_config=tile_config)

    mazes = []
    games = []
    seeds = []
    for i in range(args.num_mazes):
        seed = args.seed + i
        random.seed(seed)
        maze = generator.generate()
        game = Game(maze=maze, wind_frequency=args.wind_frequency,
                    melting_path=melting_path, relative_directions=relative_directions)
        mazes.append(maze)
        games.append(game)
        seeds.append(seed)

    # ── Run inference ──
    print(f"Running inference: {args.num_mazes} mazes, max_turns={args.max_turns}, batch_size={args.batch_size}, temp={args.temperature}, action_masking={action_masking}, relative_directions={relative_directions}, melting_path={melting_path}, enable_thinking={enable_thinking}")
    t_infer = time.time()

    results = run_batched_inference(
        model, tokenizer, games, mazes, seeds,
        max_turns=args.max_turns,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        rewards_cfg=rewards_cfg,
        action_masking=action_masking,
        enable_thinking=enable_thinking,
    )

    infer_time = time.time() - t_infer
    total_time = time.time() - t_start

    # ── Save results ──
    trajectories_path = output_dir / "trajectories.json"
    with open(trajectories_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved trajectories to {trajectories_path}")

    # Save config
    config_out = {
        "checkpoint": str(checkpoint_path),
        "base_model": base_model,
        "base_model_only": args.base_model_only,
        "action_masking": action_masking,
        "relative_directions": relative_directions,
        "melting_path": melting_path,
        "enable_thinking": enable_thinking,
        "num_mazes": args.num_mazes,
        "max_turns": args.max_turns,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "wind_frequency": args.wind_frequency,
        "maze_size": maze_size,
        "goal_lava_ratio": goal_lava_ratio,
        "tile_config": {"mode": tile_config.mode, "PATH": tile_config.PATH, "LAVA": tile_config.LAVA, "GOAL": tile_config.GOAL},
        "rewards": rewards_cfg,
        "timing": {"inference_s": round(infer_time, 1), "total_s": round(total_time, 1)},
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"Saved config to {config_path}")

    # ── Print summary ──
    print_summary(results)
    print(f"\nTiming: inference={infer_time:.1f}s  total={total_time:.1f}s")


if __name__ == "__main__":
    main()
