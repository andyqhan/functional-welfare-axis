"""Batched maze rollout engine with KV-cache acceleration."""

import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ..maze.game import Game
from ..maze.maze import MazeGenerator
from ..maze.tiles import TileConfig
from .config import TrainConfig
from .grpo import compute_action_entropy

DIRECTIONS = {"N", "E", "S", "W"}
DIRECTION_NAMES = ["north", "east", "south", "west"]  # matches N, E, S, W token order
DIRECTION_WORDS = {"North": "N", "South": "S", "East": "E", "West": "W"}


@dataclass
class TurnTokens:
    """Chat template turn structure tokens, derived from tokenizer."""
    turn_prefix: list[int]   # <|im_end|>\n<|im_start|>user\n
    turn_suffix: list[int]   # <|im_end|>\n<|im_start|>assistant\n
    think_prefix: list[int]  # <think>\n\n</think>\n\n (thinking models) or [] (non-thinking)
    im_end_tokens: list[int] # <|im_end|>\n


def model_supports_thinking(tokenizer) -> bool:
    """Check if the model's chat template supports the thinking parameter.

    Returns True for models like Qwen3-8B where enable_thinking=False produces
    different output (adds empty think block). Returns False for models like
    Qwen3-4B-Instruct where the parameter has no effect or is unsupported.
    """
    test_msgs = [{"role": "user", "content": "Hi"}]
    try:
        default_ids = tokenizer.apply_chat_template(
            test_msgs, add_generation_prompt=True, tokenize=True,
        )
        no_think_ids = tokenizer.apply_chat_template(
            test_msgs, add_generation_prompt=True, tokenize=True,
            enable_thinking=False,
        )
        return default_ids != no_think_ids
    except TypeError:
        # Template doesn't accept enable_thinking parameter
        return False


def derive_turn_tokens(tokenizer, enable_thinking: bool = False) -> TurnTokens:
    """Derive turn structure tokens from the tokenizer's chat template.

    Instead of hardcoding token IDs, computes them by diffing 1-turn vs 3-message
    template outputs. Also derives think_prefix for thinking models in no-thinking mode.

    Args:
        tokenizer: HF tokenizer with chat template.
        enable_thinking: config.enable_thinking value. When False and model supports
            thinking, injects empty think block tokens.

    Returns:
        TurnTokens with derived token sequences.
    """
    # Use single-char content that tokenizes to 1 token each
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    assert len(a_ids) == 1 and len(b_ids) == 1, (
        f"Marker content must be single tokens, got A={a_ids}, B={b_ids}"
    )

    # 1-message: [user: "A"] with generation prompt
    msgs_1 = [{"role": "user", "content": "A"}]
    ids_1 = tokenizer.apply_chat_template(
        msgs_1, add_generation_prompt=True, tokenize=True,
    )

    # 3-message: [user: "A", assistant: "B", user: "A"] with generation prompt
    msgs_3 = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "A"},
    ]
    ids_3 = tokenizer.apply_chat_template(
        msgs_3, add_generation_prompt=True, tokenize=True,
    )

    # Extract turn tokens from the tail: B + turn_prefix + A + turn_suffix
    tail = ids_3[len(ids_1):]
    assert tail[:len(b_ids)] == b_ids, (
        f"Expected B tokens at start of tail, got {tail[:len(b_ids)]}"
    )
    remaining = tail[len(b_ids):]

    # Find A tokens in remaining to split turn_prefix and turn_suffix
    a_start = None
    for i in range(len(remaining) - len(a_ids) + 1):
        if remaining[i:i + len(a_ids)] == a_ids:
            a_start = i
            break
    assert a_start is not None, "Could not find A tokens in template tail"

    turn_prefix = remaining[:a_start]
    turn_suffix = remaining[a_start + len(a_ids):]

    # Derive im_end_tokens = [<|im_end|>, \n] from the tokenizer directly.
    # Can't use apply_chat_template on a complete conversation because thinking
    # models (8B) add think tokens that break the offset calculation.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id is not None and im_end_id != getattr(tokenizer, 'unk_token_id', None), (
        "Could not resolve <|im_end|> token ID"
    )
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline_ids) == 1, f"Newline tokenizes to {len(newline_ids)} tokens"
    im_end_tokens = [im_end_id, newline_ids[0]]
    # Verify: both turn_prefix and turn_suffix should start with im_end_tokens
    assert turn_prefix[:2] == im_end_tokens, (
        f"turn_prefix doesn't start with im_end_tokens: {turn_prefix[:2]} vs {im_end_tokens}"
    )
    assert turn_suffix[:2] == im_end_tokens, (
        f"turn_suffix doesn't start with im_end_tokens: {turn_suffix[:2]} vs {im_end_tokens}"
    )

    # Derive think_prefix for thinking models in no-thinking mode
    think_prefix = []
    if not enable_thinking and model_supports_thinking(tokenizer):
        ids_no_think = tokenizer.apply_chat_template(
            msgs_1, add_generation_prompt=True, tokenize=True,
            enable_thinking=False,
        )
        think_prefix = ids_no_think[len(ids_1):]
        assert len(think_prefix) > 0, (
            "Model supports thinking but no think prefix tokens found"
        )

    return TurnTokens(
        turn_prefix=turn_prefix,
        turn_suffix=turn_suffix,
        think_prefix=think_prefix,
        im_end_tokens=im_end_tokens,
    )


def derive_turn_tokens_plain(tokenizer) -> TurnTokens:
    """Derive turn structure tokens for plain-text (non-ChatML) format.

    Plain-text format uses:
        <user prompt>\n\nI move: <direction>\n\n<user prompt>\n\nI move: ...

    Args:
        tokenizer: HF tokenizer.

    Returns:
        TurnTokens with plain-text token sequences.
    """
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    assert len(a_ids) == 1 and len(b_ids) == 1, (
        f"Marker content must be single tokens, got A={a_ids}, B={b_ids}"
    )

    # turn_prefix: \n\n (separator after direction, before next maze prompt)
    turn_prefix = tokenizer.encode("\n\n", add_special_tokens=False)
    # turn_suffix: \n\nI move:  (after maze prompt, before generation — note trailing space)
    turn_suffix = tokenizer.encode("\n\nI move: ", add_special_tokens=False)

    # Round-trip assertion: encode "A\n\nB\n\nI move: " and verify composition
    full_text = "A\n\nB\n\nI move: "
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    expected = a_ids + turn_prefix + b_ids + turn_suffix
    assert full_ids == expected, (
        f"Plain-text turn token round-trip failed: "
        f"encode('{full_text}')={full_ids} != {expected} "
        f"(a={a_ids}, prefix={turn_prefix}, b={b_ids}, suffix={turn_suffix})"
    )

    return TurnTokens(
        turn_prefix=turn_prefix,
        turn_suffix=turn_suffix,
        think_prefix=[],
        im_end_tokens=[],  # no im_end in plain-text format
    )


@dataclass
class Trajectory:
    """Data for a single rollout trajectory."""
    prompt_idx: int  # which prompt group (0..num_prompts-1)
    messages: list = field(default_factory=list)
    response_token_ids: list = field(default_factory=list)
    old_log_probs: list = field(default_factory=list)
    response_positions: list = field(default_factory=list)  # position in full_ids
    action_entropies: list = field(default_factory=list)  # per-turn entropy over direction tokens (nats)
    tile_types: list = field(default_factory=list)  # per-turn [4] ints: 0=OOB, 1=PATH, 2=LAVA, 3=GOAL
    per_turn_rewards: list = field(default_factory=list)  # per-turn reward signal
    thinking_token_counts: list = field(default_factory=list)  # per-turn thinking token count (thinking rollout only)
    is_direction_position: list = field(default_factory=list)  # per-response-position: True at direction tokens, False at thinking tokens
    full_token_ids: list = field(default_factory=list)  # complete token sequence (built incrementally)
    num_turns: int = 0
    lava_visits: int = 0
    goal_visits: int = 0
    player_trajectory: list = field(default_factory=list)
    active: bool = True
    # Termination reason: "max_turns" (ran all turns), "invalid_output" (non-direction token),
    # "invalid_move" (direction rejected by game), or None (still active)
    termination_reason: str | None = None
    invalid_output_text: str | None = None  # the text that caused termination


def generate_mazes(config: TrainConfig, step: int, rank: int = 0, world_size: int = 1):
    """Generate a batch of mazes for this training step.

    In distributed mode, each rank generates its share of mazes with
    globally unique seeds. Rank 0 of N-GPU generates the same mazes
    as the first 1/N of single-GPU for reproducibility.
    """
    assert config.num_prompts % world_size == 0, (
        f"num_prompts ({config.num_prompts}) must be divisible by world_size ({world_size})"
    )

    tile_config = TileConfig.from_custom_config({
        "tile_mode": config.tile_mode,
        "tile_chars": config.tile_chars,
    })
    generator = MazeGenerator(
        size=config.maze_size,
        goal_lava_ratio=config.goal_lava_ratio,
        tile_config=tile_config,
    )

    local_num_prompts = config.num_prompts // world_size
    start_idx = rank * local_num_prompts

    mazes = []
    for i in range(local_num_prompts):
        seed = config.base_seed + step * config.num_prompts + (start_idx + i)
        random.seed(seed)
        maze = generator.generate()
        mazes.append(maze)
    return mazes


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample from logits with temperature and nucleus (top-p) sampling.

    Args:
        logits: [batch_size, vocab_size] raw logits.
        temperature: sampling temperature (>0).
        top_p: nucleus sampling threshold (0-1).

    Returns:
        sampled_tokens: [batch_size] sampled token IDs.
    """
    assert temperature > 0, f"temperature must be > 0, got {temperature}"
    assert 0 < top_p <= 1, f"top_p must be in (0, 1], got {top_p}"

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Zero out tokens above the top-p threshold
    sorted_mask = (cumulative_probs - sorted_probs) > top_p
    sorted_probs[sorted_mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample
    sampled_idx = torch.multinomial(sorted_probs, 1)  # [bs, 1]
    sampled_token = sorted_indices.gather(-1, sampled_idx)  # [bs, 1]

    return sampled_token.squeeze(-1)


def sample_from_directions(
    logits: torch.Tensor,
    direction_token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Sample from only the 4 direction tokens {N, E, S, W}.

    Extracts logits for the 4 direction tokens, applies temperature,
    and samples from the resulting categorical distribution.

    Args:
        logits: [batch_size, vocab_size] raw logits.
        direction_token_ids: [4] token IDs for N, E, S, W.
        temperature: sampling temperature (>0).

    Returns:
        sampled_tokens: [batch_size] sampled token IDs (always a direction).
    """
    assert temperature > 0, f"temperature must be > 0, got {temperature}"
    dir_logits = logits[:, direction_token_ids]  # [batch_size, 4]
    dir_logits = dir_logits / temperature
    probs = F.softmax(dir_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1).squeeze(1)  # [batch_size]
    return direction_token_ids[sampled_idx]


def _resolve_direction_tokens(tokenizer, device: torch.device) -> torch.Tensor:
    """Resolve N/E/S/W to their token IDs. Returns tensor of shape [4]."""
    nesw_token_ids = []
    for d in ["N", "E", "S", "W"]:
        ids = tokenizer.encode(d, add_special_tokens=False)
        assert len(ids) == 1, f"Direction '{d}' tokenizes to {len(ids)} tokens: {ids}"
        nesw_token_ids.append(ids[0])
    result = torch.tensor(nesw_token_ids, device=device)
    assert len(set(nesw_token_ids)) == 4, f"Direction tokens must be distinct, got {nesw_token_ids}"
    return result


def pad_left(sequences: list[list[int]], pad_token_id: int) -> list[list[int]]:
    """Left-pad token ID sequences to uniform length."""
    max_len = max(len(s) for s in sequences)
    return [[pad_token_id] * (max_len - len(s)) + s for s in sequences]


def compute_reward(t: Trajectory, config: TrainConfig) -> float:
    """Compute maze reward from trajectory data."""
    steps = len(t.player_trajectory)  # includes start position
    reward = config.step_penalty * steps
    reward += config.goal_reward * t.goal_visits
    reward += config.lava_penalty * t.lava_visits
    return reward


def _encode_tile_types(game: Game) -> list[int]:
    """Encode tile types for each direction as integers.

    Returns [4] ints in N/E/S/W order: 0=OOB (out of bounds), 1=PATH, 2=LAVA, 3=GOAL.
    """
    available = game.get_available_directions()
    tc = game.maze.tile_config
    tile_to_int = {tc.PATH: 1, tc.LAVA: 2, tc.GOAL: 3}
    return [tile_to_int.get(available.get(d), 0) for d in DIRECTION_NAMES]


@torch.no_grad()
def run_rollout(
    model: torch.nn.Module,
    tokenizer,
    mazes: list,
    config: TrainConfig,
    device: torch.device,
):
    """Run batched maze rollouts with KV-cache acceleration.

    Processes each micro-batch independently through ALL turns, keeping
    KV cache memory bounded to one micro-batch at a time. At each turn:
    1. Sample 1 token from cached logits
    2. Decode direction, update game state
    3. Tokenize new conversation tokens (assistant response + env feedback)
    4. Forward only new tokens with KV cache → logits at last position

    This avoids recomputing the full sequence (~1300 tokens) every turn.
    Instead, only ~70 new tokens/turn are processed, giving ~18x speedup.

    Args:
        model: actor model (gradient checkpointing will be temporarily disabled).
        tokenizer: HF tokenizer.
        mazes: list of Maze objects (len = num_prompts).
        config: training configuration.
        device: torch device.

    Returns:
        (trajectories, rollout_timing): list of Trajectory objects + timing dict.
    """
    # Disable gradient checkpointing for inference (enables use_cache)
    model.gradient_checkpointing_disable()
    model.eval()

    # Derive turn structure tokens from template (replaces hardcoded constants)
    if config.use_chat_template:
        turn_tokens = derive_turn_tokens(tokenizer, enable_thinking=config.enable_thinking)
    else:
        turn_tokens = derive_turn_tokens_plain(tokenizer)

    # Always resolve direction token IDs (needed for entropy; also used for action masking)
    nesw_token_ids = _resolve_direction_tokens(tokenizer, device)

    num_prompts = len(mazes)
    total = num_prompts * config.group_size

    # Initialize all trajectories and games
    trajectories: list[Trajectory] = []
    games: list[Game] = []

    for prompt_idx, maze in enumerate(mazes):
        # Compute direction order once per prompt (shared across GRPO group)
        if config.shuffle_directions:
            direction_order = random.sample(["north", "east", "south", "west"], 4)
        else:
            direction_order = None  # Game defaults to standard order

        for _ in range(config.group_size):
            game = Game(
                maze,
                wind_frequency=config.wind_frequency,
                melting_path=config.melting_path,
                direction_order=direction_order,
                relative_directions=config.relative_directions,
            )
            initial_prompt = game.get_prompt()
            t = Trajectory(
                prompt_idx=prompt_idx,
                messages=[{"role": "user", "content": initial_prompt}],
                player_trajectory=[game.position],
            )
            trajectories.append(t)
            games.append(game)

    assert len(trajectories) == total

    rollout_timing = {"tokenize": 0.0, "forward": 0.0, "sample": 0.0, "update": 0.0}
    mb_size = config.rollout_micro_batch_size

    # Process each micro-batch independently through all turns
    for mb_start in range(0, total, mb_size):
        mb_end = min(mb_start + mb_size, total)
        mb_trajs = trajectories[mb_start:mb_end]
        mb_games = games[mb_start:mb_end]
        mb_bs = mb_end - mb_start

        # --- Tokenize initial prompts ---
        t0 = time.time()
        initial_ids_list = []
        for t in mb_trajs:
            if config.use_chat_template:
                ids = tokenizer.apply_chat_template(
                    t.messages, add_generation_prompt=True, tokenize=True,
                )
            else:
                # Plain-text: "<user content>\n\nI move: "
                initial_text = t.messages[0]["content"] + "\n\nI move: "
                ids = tokenizer.encode(initial_text, add_special_tokens=False)
            initial_ids_list.append(ids)
            # Initialize full_token_ids with initial prompt + think_prefix
            t.full_token_ids = list(ids) + list(turn_tokens.think_prefix)
        # Track unpadded conversation length per trajectory (for response_positions)
        # Includes think_prefix since those tokens precede the first direction
        content_lens = [len(ids) + len(turn_tokens.think_prefix) for ids in initial_ids_list]
        rollout_timing["tokenize"] += time.time() - t0

        # --- Prefill: forward pass on full initial prompts ---
        t0 = time.time()
        padded = pad_left(initial_ids_list, tokenizer.pad_token_id)
        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = (input_tensor != tokenizer.pad_token_id).long()
        cache_len = input_tensor.shape[1]  # includes left-padding

        cache = DynamicCache()
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        last_logits = outputs.logits[:, -1, :]  # [mb_bs, vocab_size]
        del outputs

        # Forward think_prefix tokens through KV cache (thinking models only)
        if turn_tokens.think_prefix:
            think_ids = torch.tensor(
                [turn_tokens.think_prefix] * mb_bs, dtype=torch.long, device=device,
            )
            think_mask = torch.ones(mb_bs, len(turn_tokens.think_prefix), dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, think_mask], dim=1)
            cache_position = torch.arange(
                cache_len, cache_len + len(turn_tokens.think_prefix), device=device,
            )
            outputs = model(
                input_ids=think_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            last_logits = outputs.logits[:, -1, :]  # [mb_bs, vocab_size]
            cache_len += len(turn_tokens.think_prefix)
            del outputs

        rollout_timing["forward"] += time.time() - t0

        # Track which trajectories in this micro-batch are still active
        active = [True] * mb_bs

        for turn in range(config.max_turns):
            if not any(active):
                break

            # --- Sample tokens ---
            t0 = time.time()
            if config.action_masking:
                sampled_tokens = sample_from_directions(last_logits, nesw_token_ids, config.temperature)
            else:
                sampled_tokens = sample_top_p(last_logits, config.temperature, config.top_p)
            token_log_probs = F.log_softmax(last_logits, dim=-1)
            old_log_probs = token_log_probs.gather(
                1, sampled_tokens.unsqueeze(1),
            ).squeeze(1)

            # Compute entropy over direction tokens (raw logits, no temperature)
            turn_entropy = compute_action_entropy(last_logits, nesw_token_ids)
            for i in range(mb_bs):
                if active[i]:
                    mb_trajs[i].action_entropies.append(turn_entropy[i].item())

            sampled_tokens_cpu = sampled_tokens.cpu()
            old_log_probs_cpu = old_log_probs.cpu()
            rollout_timing["sample"] += time.time() - t0

            # --- Capture tile types BEFORE moves (for equalized entropy) ---
            for i in range(mb_bs):
                if active[i]:
                    mb_trajs[i].tile_types.append(_encode_tile_types(mb_games[i]))

            # --- Update game states (only for active trajectories) ---
            t0 = time.time()
            for i in range(mb_bs):
                if not active[i]:
                    continue

                t = mb_trajs[i]
                game = mb_games[i]
                token_id = sampled_tokens_cpu[i].item()
                direction_text = tokenizer.decode([token_id]).strip()
                t.num_turns += 1

                # Parse direction from the sampled token text.
                # Accept the first character if it's a valid direction AND
                # the next character (if any) is not lowercase — this avoids
                # misinterpreting words like "No" or "South" as directions
                # while still accepting "NNN", "N.", "N\n", etc.
                # The conversation history records only the parsed direction
                # so the model sees clean single-char responses.
                direction = None
                record_log_prob = old_log_probs_cpu[i].item()

                if direction_text in DIRECTIONS:
                    direction = direction_text
                elif direction_text in DIRECTION_WORDS:
                    direction = DIRECTION_WORDS[direction_text]
                elif (
                    direction_text
                    and direction_text[0] in DIRECTIONS
                    and (len(direction_text) < 2 or not direction_text[1].islower())
                ):
                    direction = direction_text[0]

                if direction is not None:
                    # Always re-encode to get canonical token for retokenization
                    correct_ids = tokenizer.encode(direction, add_special_tokens=False)
                    assert len(correct_ids) == 1, (
                        f"Direction '{direction}' tokenizes to {len(correct_ids)} tokens"
                    )
                    record_token_id = correct_ids[0]
                    if record_token_id != token_id:
                        record_log_prob = token_log_probs[i, record_token_id].item()
                else:
                    # Non-direction token: trajectory will be terminated.
                    # Re-encode from message content to match retokenization.
                    content = direction_text or "?"
                    correct_ids = tokenizer.encode(content, add_special_tokens=False)
                    record_token_id = correct_ids[0]
                    if record_token_id != token_id:
                        record_log_prob = token_log_probs[i, record_token_id].item()

                t.response_token_ids.append(record_token_id)
                t.old_log_probs.append(record_log_prob)
                t.response_positions.append(content_lens[i])

                if direction is None:
                    active[i] = False
                    t.active = False
                    t.termination_reason = "invalid_output"
                    t.invalid_output_text = direction_text
                    t.messages.append(
                        {"role": "assistant", "content": direction_text or "?"},
                    )
                    t.per_turn_rewards.append(config.step_penalty)
                    continue

                try:
                    prev_lava = t.lava_visits
                    prev_goal = t.goal_visits
                    game.move(direction)
                except ValueError:
                    active[i] = False
                    t.active = False
                    t.termination_reason = "invalid_move"
                    t.invalid_output_text = direction
                    t.messages.append({"role": "assistant", "content": direction})
                    t.per_turn_rewards.append(config.step_penalty)
                    continue

                # Valid move
                t.messages.append({"role": "assistant", "content": direction})
                t.messages.append(
                    {"role": "user", "content": game.get_prompt()},
                )
                t.player_trajectory.append(game.position)
                t.lava_visits = game.get_lava_visit_count()
                t.goal_visits = game.get_goal_visit_count()
                delta_goal = t.goal_visits - prev_goal
                delta_lava = t.lava_visits - prev_lava
                turn_reward = config.step_penalty + config.goal_reward * delta_goal + config.lava_penalty * delta_lava
                t.per_turn_rewards.append(turn_reward)

            rollout_timing["update"] += time.time() - t0

            if not any(active):
                break

            # --- Tokenize new tokens for active trajectories (incremental) ---
            # Instead of retokenizing the full conversation each turn, build
            # new tokens from: direction_token + template + user_content + template + think_prefix
            t0 = time.time()
            new_tokens_list = []
            for i in range(mb_bs):
                if not active[i]:
                    new_tokens_list.append([tokenizer.pad_token_id])
                    continue

                t = mb_trajs[i]
                # The last 2 messages are the assistant direction + user env feedback
                user_content = t.messages[-1]["content"]
                content_tokens = tokenizer.encode(
                    user_content, add_special_tokens=False,
                )
                new_tokens = (
                    [t.response_token_ids[-1]]  # direction token
                    + turn_tokens.turn_prefix
                    + content_tokens
                    + turn_tokens.turn_suffix
                    + turn_tokens.think_prefix  # empty for non-thinking models
                )
                assert len(new_tokens) > 0

                # Validate incremental tokenization against full retokenization
                # (first micro-batch, first 2 turns, first 3 trajectories)
                # Only for ChatML non-thinking models — thinking models inject think_prefix
                # that the default template doesn't include, and plain-text mode has BPE
                # boundary differences between full and incremental encoding (expected and OK
                # since rollout + training both use the same incremental tokens)
                if mb_start == 0 and turn < 2 and i < 3 and not turn_tokens.think_prefix and config.use_chat_template:
                    full_ids = tokenizer.apply_chat_template(
                        t.messages, add_generation_prompt=True, tokenize=True,
                    )
                    expected_new = full_ids[content_lens[i]:]
                    assert new_tokens == expected_new, (
                        f"Incremental tokenization mismatch at traj {i}, turn {turn}: "
                        f"incremental={new_tokens[:10]}... vs full={expected_new[:10]}... "
                        f"(lens: {len(new_tokens)} vs {len(expected_new)})"
                    )

                t.full_token_ids.extend(new_tokens)
                content_lens[i] += len(new_tokens)
                new_tokens_list.append(new_tokens)
            rollout_timing["tokenize"] += time.time() - t0

            # --- Forward pass: only new tokens with KV cache ---
            t0 = time.time()
            max_new_len = max(len(t) for t in new_tokens_list)
            new_ids = torch.full(
                (mb_bs, max_new_len), tokenizer.pad_token_id,
                dtype=torch.long, device=device,
            )
            new_mask = torch.zeros(
                mb_bs, max_new_len, dtype=torch.long, device=device,
            )
            for i, toks in enumerate(new_tokens_list):
                t_tensor = torch.tensor(toks, dtype=torch.long, device=device)
                new_ids[i, :len(toks)] = t_tensor
                new_mask[i, :len(toks)] = 1

            # Extend attention mask to cover cache + new tokens
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            cache_position = torch.arange(
                cache_len, cache_len + max_new_len, device=device,
            )

            outputs = model(
                input_ids=new_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                # Can't use logits_to_keep=1 here because right-padded sequences
                # have their last real token at different positions
            )
            cache_len += max_new_len

            # Extract logits at last REAL position for each sequence
            last_real_idx = new_mask.sum(dim=1).long() - 1  # [mb_bs]
            last_logits = outputs.logits[
                torch.arange(mb_bs, device=device), last_real_idx,
            ]  # [mb_bs, vocab_size]
            del outputs
            rollout_timing["forward"] += time.time() - t0

        # --- Finalize full_token_ids for each trajectory in this micro-batch ---
        for i in range(mb_bs):
            t = mb_trajs[i]
            if t.active:
                # Max turns reached: full_token_ids ends with turn_suffix + think_prefix
                # from the last forwarded turn. Trim trailing turn_suffix + think_prefix.
                trim_len = len(turn_tokens.turn_suffix) + len(turn_tokens.think_prefix)
                if config.use_chat_template:
                    # ChatML: replace with im_end_tokens
                    t.full_token_ids = t.full_token_ids[:-trim_len] + list(turn_tokens.im_end_tokens)
                else:
                    # Plain-text: just trim the trailing suffix (no im_end)
                    t.full_token_ids = t.full_token_ids[:-trim_len]
            else:
                # Terminated (invalid_output or invalid_move): the last direction token
                # was recorded but not added to full_token_ids (trajectory was inactive
                # when tokenize section ran). Append it (+ im_end for ChatML).
                if config.use_chat_template:
                    t.full_token_ids.extend(
                        [t.response_token_ids[-1]] + list(turn_tokens.im_end_tokens),
                    )
                else:
                    t.full_token_ids.append(t.response_token_ids[-1])

        # Free KV cache for this micro-batch
        del cache
        torch.cuda.empty_cache()

    # Re-enable gradient checkpointing for training (if it was enabled)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    # Mark trajectories that used all turns (no early termination)
    for t in trajectories:
        if t.termination_reason is None:
            t.termination_reason = "max_turns"

    # Compute rewards
    for t in trajectories:
        t.reward = compute_reward(t, config)

    return trajectories, rollout_timing


def prepare_training_batch(
    trajectories: list[Trajectory],
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert trajectories into padded tensors for the training forward pass.

    Returns dict with:
        - input_ids: [batch, max_seq_len] left-padded full token sequences
        - attention_mask: [batch, max_seq_len]
        - response_positions: [batch, max_turns] positions of response tokens
        - response_token_ids: [batch, max_turns] the sampled token IDs
        - response_mask: [batch, max_turns] 1 for valid turns, 0 for padding
        - old_log_probs: [batch, max_turns]
        - rewards: [batch]
        - group_indices: [batch] prompt group index for each trajectory
    """
    batch_size = len(trajectories)

    # Use pre-built full token sequences (avoids retokenization mismatch
    # with thinking models where apply_chat_template adds think tags
    # inconsistently to only the last assistant message)
    all_full_ids = []
    for t in trajectories:
        assert len(t.full_token_ids) > 0, "Trajectory full_token_ids not built"
        all_full_ids.append(t.full_token_ids)

    # Find max sequence length and max turns
    max_seq_len = max(len(ids) for ids in all_full_ids)
    max_turns = max(len(t.response_positions) for t in trajectories)
    assert max_turns > 0, "All trajectories have 0 turns"

    # Left-pad input sequences
    padded_ids = pad_left(all_full_ids, tokenizer.pad_token_id)
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Compute padding offset: how much left-padding was added per sequence
    pad_offsets = [max_seq_len - len(ids) for ids in all_full_ids]

    # Build response-level tensors (padded to max_turns)
    response_positions = torch.zeros(batch_size, max_turns, dtype=torch.long, device=device)
    response_token_ids = torch.zeros(batch_size, max_turns, dtype=torch.long, device=device)
    response_mask = torch.zeros(batch_size, max_turns, dtype=torch.float32, device=device)
    old_log_probs = torch.zeros(batch_size, max_turns, dtype=torch.float32, device=device)

    for i, t in enumerate(trajectories):
        n_turns = len(t.response_positions)
        assert n_turns == len(t.response_token_ids) == len(t.old_log_probs), (
            f"Trajectory {i}: positions={len(t.response_positions)}, "
            f"tokens={len(t.response_token_ids)}, log_probs={len(t.old_log_probs)}"
        )
        for j in range(n_turns):
            # Adjust position for left-padding offset
            pos = t.response_positions[j] + pad_offsets[i]
            assert 0 <= pos < max_seq_len, (
                f"Response position {pos} out of bounds [0, {max_seq_len}) "
                f"(original={t.response_positions[j]}, pad_offset={pad_offsets[i]})"
            )
            response_positions[i, j] = pos
            response_token_ids[i, j] = t.response_token_ids[j]
            response_mask[i, j] = 1.0
            old_log_probs[i, j] = t.old_log_probs[j]

    # Build direction_mask: [batch, max_response_len] — 1 at direction positions, 0 at thinking
    # When is_direction_position is empty (non-thinking rollout), all positions are directions.
    direction_mask = torch.zeros(batch_size, max_turns, dtype=torch.float32, device=device)
    for i, t in enumerate(trajectories):
        if t.is_direction_position:
            for j, is_dir in enumerate(t.is_direction_position):
                direction_mask[i, j] = 1.0 if is_dir else 0.0
        else:
            # Non-thinking rollout: all response positions are direction tokens
            for j in range(len(t.response_positions)):
                direction_mask[i, j] = 1.0

    # Build tile_types tensor: [batch, max_response_len, 4]
    # tile_types has one entry per game turn. When train_on_thinking=True,
    # response_positions has many entries per turn (thinking + direction).
    # We place each turn's tile_types at the corresponding direction position
    # in the expanded tensor. Thinking positions get [0,0,0,0].
    tile_types = torch.zeros(batch_size, max_turns, 4, dtype=torch.long, device=device)
    for i, t in enumerate(trajectories):
        if t.is_direction_position and t.tile_types:
            # Place tile_types at direction position indices
            dir_idx = 0
            for j, is_dir in enumerate(t.is_direction_position):
                if is_dir and dir_idx < len(t.tile_types):
                    tile_types[i, j] = torch.tensor(t.tile_types[dir_idx], dtype=torch.long)
                    dir_idx += 1
        else:
            # Non-thinking rollout: one tile_type per response position
            for j, tt in enumerate(t.tile_types):
                tile_types[i, j] = torch.tensor(tt, dtype=torch.long)

    # Build per_turn_rewards tensor: [batch, max_turns]
    per_turn_rewards = torch.zeros(batch_size, max_turns, dtype=torch.float32, device=device)
    for i, t in enumerate(trajectories):
        for j, r in enumerate(t.per_turn_rewards):
            per_turn_rewards[i, j] = r

    rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32, device=device)
    group_indices = torch.tensor([t.prompt_idx for t in trajectories], dtype=torch.long, device=device)

    # Validate: check that response tokens match what's in full_ids at response positions
    for i in range(min(batch_size, 10)):  # spot-check first 10
        for j in range(int(response_mask[i].sum().item())):
            pos = response_positions[i, j].item()
            expected = response_token_ids[i, j].item()
            actual = input_ids[i, pos].item()
            assert actual == expected, (
                f"Token mismatch at traj {i}, pos {j}: "
                f"input_ids[{i},{pos}]={actual} != response_token_id={expected}. "
                f"Token text: actual='{tokenizer.decode([actual])}', expected='{tokenizer.decode([expected])}'"
            )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_positions": response_positions,
        "response_token_ids": response_token_ids,
        "response_mask": response_mask,
        "direction_mask": direction_mask,
        "old_log_probs": old_log_probs,
        "per_turn_rewards": per_turn_rewards,
        "rewards": rewards,
        "group_indices": group_indices,
        "tile_types": tile_types,
    }


def compute_log_probs_at_positions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_positions: torch.Tensor,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    micro_batch_size: int,
    direction_token_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities of response tokens via batched forward passes.

    Uses logits_to_keep to only compute logits at the positions we need,
    reducing memory from [batch, seq_len, vocab] to [batch, ~max_turns, vocab].

    Args:
        model: the language model (actor or reference).
        input_ids: [batch, seq_len] padded token sequences.
        attention_mask: [batch, seq_len].
        response_positions: [batch, max_turns] positions of response tokens.
        response_token_ids: [batch, max_turns] the response token IDs.
        response_mask: [batch, max_turns] valid turn mask.
        micro_batch_size: batch size per forward pass.
        direction_token_ids: [4] token IDs for N/E/S/W. If provided, also returns
            direction logits at each response position.

    Returns:
        If direction_token_ids is None:
            log_probs: [batch, max_turns]
        If direction_token_ids is provided:
            (log_probs, direction_logits): [batch, max_turns] and [batch, max_turns, 4]
    """
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    max_turns = response_positions.shape[1]
    device = input_ids.device

    all_log_probs = torch.zeros(batch_size, max_turns, dtype=torch.float32, device=device)
    if direction_token_ids is not None:
        all_direction_logits = torch.zeros(batch_size, max_turns, 4, dtype=torch.float32, device=device)

    for mb_start in range(0, batch_size, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, batch_size)
        mb_input_ids = input_ids[mb_start:mb_end]
        mb_attention_mask = attention_mask[mb_start:mb_end]
        mb_positions = response_positions[mb_start:mb_end]
        mb_token_ids = response_token_ids[mb_start:mb_end]
        mb_mask = response_mask[mb_start:mb_end]
        mb_size = mb_end - mb_start

        # logits[i, p-1, :] predicts token at position p
        pred_positions = mb_positions - 1  # [mb_size, max_turns]
        pred_positions = pred_positions.clamp(min=0)

        # Build a boolean mask of which positions need logits (union across batch)
        # This reduces logits from [mb_size, seq_len, vocab] to [mb_size, ~N, vocab]
        pos_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        valid_pos = pred_positions[mb_mask.bool()]
        if valid_pos.numel() > 0:
            pos_mask[valid_pos.unique()] = True
        else:
            # No valid positions, skip this micro-batch
            continue

        # Forward pass with selective logits computation
        outputs = model(
            input_ids=mb_input_ids,
            attention_mask=mb_attention_mask,
            logits_to_keep=pos_mask,
        )
        selected_logits = outputs.logits  # [mb_size, num_selected, vocab_size]

        # Map from original positions to indices in selected logits
        selected_positions = pos_mask.nonzero(as_tuple=True)[0]  # [num_selected]
        pos_to_selected_idx = torch.zeros(seq_len, dtype=torch.long, device=device)
        pos_to_selected_idx[selected_positions] = torch.arange(
            len(selected_positions), device=device,
        )

        # Gather logits at prediction positions (now mapped to selected indices)
        pred_selected_idx = pos_to_selected_idx[pred_positions]  # [mb_size, max_turns]
        batch_idx = torch.arange(mb_size, device=device).unsqueeze(1).expand(-1, max_turns)
        pred_logits = selected_logits[batch_idx, pred_selected_idx]  # [mb_size, max_turns, vocab]

        # Compute log probs and extract for target tokens
        log_probs = F.log_softmax(pred_logits, dim=-1)  # [mb_size, max_turns, vocab_size]
        token_log_probs = log_probs.gather(-1, mb_token_ids.unsqueeze(-1)).squeeze(-1)

        # Apply mask (set padded positions to 0)
        token_log_probs = token_log_probs * mb_mask

        all_log_probs[mb_start:mb_end] = token_log_probs

        # Extract direction logits if requested
        if direction_token_ids is not None:
            dir_logits = pred_logits[..., direction_token_ids]  # [mb_size, max_turns, 4]
            all_direction_logits[mb_start:mb_end] = dir_logits

    if direction_token_ids is not None:
        return all_log_probs, all_direction_logits
    return all_log_probs
