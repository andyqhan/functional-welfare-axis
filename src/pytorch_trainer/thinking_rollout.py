"""vLLM-based thinking rollout for reasoning models.

Single-phase generation per turn: vLLM generates the full response
(thinking + direction) in one shot. The direction is parsed from the
output — no action masking during generation (vLLM's allowed_token_ids
is broken for mxfp4-quantized models). Action masking is applied only
in the PyTorch old_log_probs computation after the rollout.

Supports Qwen3 (<think>...</think>) and gpt-oss-20b (Harmony analysis channel).
"""
from __future__ import annotations

import gc
import logging
import random
import time

import torch

from ..maze.game import Game
from .config import TrainConfig
from .renderer import ThinkingRenderer, create_renderer
from .rollout import (
    DIRECTION_NAMES,
    DIRECTION_WORDS,
    DIRECTIONS,
    Trajectory,
    _encode_tile_types,
    _resolve_direction_tokens,
    compute_reward,
)

logger = logging.getLogger(__name__)


def _parse_direction(text: str) -> str | None:
    """Parse a direction from decoded token text."""
    text = text.strip()
    if not text:
        return None
    if text in DIRECTIONS:
        return text
    if text in DIRECTION_WORDS:
        return DIRECTION_WORDS[text]
    if text[0] in DIRECTIONS and (len(text) < 2 or not text[1].islower()):
        return text[0]
    return None


def _get_child_pids() -> set[int]:
    """Get PIDs of all child processes (for vLLM subprocess tracking)."""
    import os
    my_pid = os.getpid()
    pids = set()
    for entry in os.listdir("/proc"):
        try:
            stat = open(f"/proc/{int(entry)}/stat").read()
            ppid = int(stat.split(")")[1].split()[1])
            if ppid == my_pid:
                pids.add(int(entry))
        except (ValueError, OSError, IndexError):
            pass
    return pids


def _destroy_vllm(llm, pre_vllm_pids: set[int]) -> float:
    """Destroy vLLM engine and kill its subprocesses. Returns time taken."""
    t0 = time.time()
    engine_pids = _get_child_pids() - pre_vllm_pids

    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'shutdown'):
            llm.llm_engine.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass

    del llm
    gc.collect()

    import os
    import signal
    for pid in engine_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to vLLM child pid=%d", pid)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in engine_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)

    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    logger.info("vLLM destroyed in %.1fs, killed %d child processes", elapsed, len(engine_pids))
    return elapsed


@torch.no_grad()
def run_thinking_rollout(
    model_path: str,
    lora_adapter_path: str | None,
    tokenizer,
    mazes: list,
    config: TrainConfig,
    device: torch.device,
) -> tuple[list[Trajectory], dict]:
    """Run batched maze rollouts using vLLM for thinking generation.

    Single-phase: vLLM generates the full response (thinking + bridge +
    direction + stop) in one call. The direction is the last generated token
    (vLLM strips the stop token). No action masking during generation —
    old_log_probs are computed via PyTorch after the rollout with proper masking.

    Args:
        model_path: HuggingFace model path.
        lora_adapter_path: Path to LoRA adapter directory, or None.
        tokenizer: HF tokenizer.
        mazes: List of Maze objects (len = num_prompts).
        config: Training configuration.
        device: Target device.

    Returns:
        (trajectories, timing): List of Trajectory objects + timing dict.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    timing = {
        "vllm_init": 0.0,
        "generate": 0.0,
        "game_update": 0.0,
        "tokenize": 0.0,
        "vllm_destroy": 0.0,
    }

    # --- Setup ---
    renderer = create_renderer(tokenizer, config)
    nesw_ids = _resolve_direction_tokens(tokenizer, torch.device("cpu")).tolist()
    nesw_set = set(nesw_ids)
    thinking_prefill = renderer.get_thinking_prefill()
    thinking_stop_ids = renderer.get_thinking_stop_ids()
    response_stop_ids = renderer.get_response_stop_ids()
    direction_bridge = renderer.get_direction_bridge()
    recovery_tokens = renderer.get_truncation_recovery_tokens()  # None for gpt-oss

    # Combined stop: stop at response end (e.g. <|im_end|> or <|return|>).
    # The model generates: thinking → thinking_stop → bridge → direction → response_stop.
    # vLLM strips the response_stop, so the last token in output is the direction.
    all_stop_ids = list(set(response_stop_ids))

    pre_vllm_pids = _get_child_pids()

    # --- Initialize vLLM ---
    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        max_model_len=config.vllm_max_model_len,
        enable_prefix_caching=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    if lora_adapter_path:
        llm_kwargs.update(enable_lora=True, max_lora_rank=config.lora_rank)
    llm = LLM(**llm_kwargs)
    lora_req = LoRARequest("maze_lora", 1, lora_adapter_path) if lora_adapter_path else None
    timing["vllm_init"] = time.time() - t0
    logger.info("vLLM initialized in %.1fs", timing["vllm_init"])

    # --- Initialize trajectories and games ---
    num_prompts = len(mazes)
    total = num_prompts * config.group_size
    trajectories: list[Trajectory] = []
    games: list[Game] = []

    for prompt_idx, maze in enumerate(mazes):
        direction_order = (
            random.sample(["north", "east", "south", "west"], 4)
            if config.shuffle_directions else None
        )
        for _ in range(config.group_size):
            game = Game(
                maze,
                wind_frequency=config.wind_frequency,
                melting_path=config.melting_path,
                direction_order=direction_order,
                relative_directions=config.relative_directions,
            )
            t = Trajectory(
                prompt_idx=prompt_idx,
                messages=[{"role": "user", "content": game.get_prompt()}],
                player_trajectory=[game.position],
            )
            trajectories.append(t)
            games.append(game)

    # Render initial prompts
    t0 = time.time()
    for t in trajectories:
        t.full_token_ids = list(renderer.render_initial_prompt(t.messages))
    timing["tokenize"] += time.time() - t0

    active = [True] * total
    n_invalid_output = 0
    n_recovered = 0
    n_total_turns = 0

    # --- Multi-turn loop ---
    for turn in range(config.max_turns):
        if not any(active):
            break

        active_indices = [i for i in range(total) if active[i]]

        # Record tile types before move
        for i in active_indices:
            trajectories[i].tile_types.append(_encode_tile_types(games[i]))

        # === Generate full response (thinking + direction) in one shot ===
        t0 = time.time()
        prompts = []
        for i in active_indices:
            prompt_ids = trajectories[i].full_token_ids + thinking_prefill
            prompts.append({"prompt_token_ids": prompt_ids})

        gen_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_thinking_tokens,
            stop_token_ids=all_stop_ids,
        )
        outputs = llm.generate(prompts, sampling_params=gen_params, lora_request=lora_req)
        timing["generate"] += time.time() - t0

        # === Parse outputs, recover truncated thinking, update game state ===
        t0 = time.time()
        stop_set = set(all_stop_ids)

        # First pass: parse directions, collect truncated trajectories for recovery
        parsed = {}  # idx_in_active -> (direction, gen_ids) or None
        needs_recovery = []  # (idx_in_active, i, gen_ids) for truncated trajectories

        for idx_in_active, i in enumerate(active_indices):
            output = outputs[idx_in_active].outputs[0]
            gen_ids = list(output.token_ids)
            trajectories[i].num_turns += 1
            n_total_turns += 1

            # vLLM V1 INCLUDES stop_token_ids in output.
            # Clean stop: output ends with {direction}{stop_token}
            direction = None
            if len(gen_ids) >= 2 and gen_ids[-1] in stop_set:
                direction_text = tokenizer.decode([gen_ids[-2]]).strip()
                direction = _parse_direction(direction_text)
                if direction is not None:
                    gen_ids = gen_ids[:-1]  # strip stop token

            if direction is not None:
                parsed[idx_in_active] = (direction, gen_ids)
            elif recovery_tokens is not None and output.finish_reason == "length":
                # Truncated thinking — attempt recovery by force-closing the think block
                needs_recovery.append((idx_in_active, i, gen_ids))
            else:
                parsed[idx_in_active] = None  # unrecoverable

        # Recovery pass: batch-generate 1 direction token for truncated trajectories
        if needs_recovery:
            recovery_prompts = []
            for idx_in_active, i, gen_ids in needs_recovery:
                # Prompt = original context + thinking_prefill + truncated_thinking + recovery + bridge
                prompt_ids = (
                    trajectories[i].full_token_ids
                    + thinking_prefill
                    + gen_ids
                    + recovery_tokens
                )
                recovery_prompts.append({"prompt_token_ids": prompt_ids})

            recovery_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=1,
                stop_token_ids=all_stop_ids,
                logprobs=20,  # top-20 for action masking
            )
            recovery_outputs = llm.generate(
                recovery_prompts, sampling_params=recovery_params, lora_request=lora_req,
            )

            for j, (idx_in_active, i, gen_ids) in enumerate(needs_recovery):
                rec_out = recovery_outputs[j].outputs[0]

                # Action masking: pick the direction token with highest logprob
                # from the top-20 logprobs returned by vLLM
                best_logprob = float("-inf")
                best_token_id = None
                if rec_out.logprobs and len(rec_out.logprobs) > 0:
                    top_logprobs = rec_out.logprobs[0]  # {token_id: Logprob}
                    for nesw_id in nesw_ids:
                        if nesw_id in top_logprobs:
                            lp = top_logprobs[nesw_id].logprob
                            if lp > best_logprob:
                                best_logprob = lp
                                best_token_id = nesw_id

                if best_token_id is not None:
                    direction = tokenizer.decode([best_token_id]).strip()
                    recovered_gen_ids = gen_ids + recovery_tokens + [best_token_id]
                    parsed[idx_in_active] = (direction, recovered_gen_ids)
                    n_recovered += 1
                else:
                    # None of N/E/S/W in top-20 (very unlikely)
                    parsed[idx_in_active] = None

        # Second pass: update game state using parsed results
        for idx_in_active, i in enumerate(active_indices):
            t = trajectories[i]
            game = games[i]
            result = parsed[idx_in_active]

            if result is None:
                # Unrecoverable: no valid direction
                output = outputs[idx_in_active].outputs[0]
                gen_ids = list(output.token_ids)
                active[i] = False
                t.active = False
                t.termination_reason = "invalid_output"
                t.invalid_output_text = tokenizer.decode([gen_ids[-1]]).strip() if gen_ids else "<empty>"
                t.per_turn_rewards.append(config.step_penalty)
                t.thinking_token_counts.append(len(gen_ids))
                n_invalid_output += 1

                placeholder_dir = random.choice(["N", "E", "S", "W"])
                placeholder_id = tokenizer.encode(placeholder_dir, add_special_tokens=False)[0]
                base_pos = len(t.full_token_ids) + len(thinking_prefill)
                if config.train_on_thinking:
                    # Record all sampled thinking tokens (gen_ids) + placeholder direction
                    for k, tok_id in enumerate(gen_ids):
                        t.response_positions.append(base_pos + k)
                        t.response_token_ids.append(tok_id)
                        t.old_log_probs.append(0.0)
                        t.is_direction_position.append(False)
                    # Placeholder direction token
                    t.response_positions.append(base_pos + len(gen_ids))
                    t.response_token_ids.append(placeholder_id)
                    t.old_log_probs.append(0.0)
                    t.is_direction_position.append(True)
                else:
                    t.response_positions.append(base_pos + len(gen_ids))
                    t.response_token_ids.append(placeholder_id)
                    t.old_log_probs.append(0.0)
                t.messages.append({"role": "assistant", "content": placeholder_dir})
                t.full_token_ids.extend(thinking_prefill + gen_ids + [placeholder_id] + renderer.build_finalization())
                continue

            direction, gen_ids = result

            # Valid direction found — reconstruct the token sequence
            canonical_id = tokenizer.encode(direction, add_special_tokens=False)[0]

            # Split gen_ids: everything before the last token is thinking + structural tokens,
            # the last token is the direction.
            response_tokens = gen_ids[:-1]

            # Record response positions in full_token_ids
            base_pos = len(t.full_token_ids) + len(thinking_prefill)
            if config.train_on_thinking:
                # Record ALL sampled tokens: thinking content + direction
                # gen_ids contains only tokens vLLM sampled (not thinking_prefill).
                # For recovered trajectories, gen_ids includes recovery_tokens which
                # were force-fed — those are handled in the recovery parsing above
                # by rebuilding gen_ids as: original_gen_ids + recovery_tokens + [direction].
                # We must NOT train on the recovery_tokens portion.
                if recovery_tokens is not None and len(gen_ids) > len(recovery_tokens) + 1:
                    # Check if this was a recovered trajectory by seeing if recovery_tokens
                    # appear before the direction token
                    rt_len = len(recovery_tokens)
                    potential_recovery = gen_ids[-(rt_len + 1):-1]
                    is_recovered = (potential_recovery == recovery_tokens)
                else:
                    is_recovered = False

                if is_recovered:
                    # Recovered: gen_ids = sampled_thinking + recovery_tokens + [direction]
                    # Train on sampled_thinking and direction, skip recovery_tokens
                    rt_len = len(recovery_tokens)
                    sampled_thinking = gen_ids[:-(rt_len + 1)]
                    for k, tok_id in enumerate(sampled_thinking):
                        t.response_positions.append(base_pos + k)
                        t.response_token_ids.append(tok_id)
                        t.old_log_probs.append(0.0)
                        t.is_direction_position.append(False)
                    # Direction token (after recovery_tokens in the sequence)
                    direction_pos = base_pos + len(gen_ids) - 1
                    t.response_positions.append(direction_pos)
                    t.response_token_ids.append(canonical_id)
                    t.old_log_probs.append(0.0)
                    t.is_direction_position.append(True)
                else:
                    # Normal: gen_ids = sampled_thinking + [direction]
                    for k, tok_id in enumerate(gen_ids[:-1]):
                        t.response_positions.append(base_pos + k)
                        t.response_token_ids.append(tok_id)
                        t.old_log_probs.append(0.0)
                        t.is_direction_position.append(False)
                    # Direction token
                    direction_pos = base_pos + len(response_tokens)
                    t.response_positions.append(direction_pos)
                    t.response_token_ids.append(canonical_id)
                    t.old_log_probs.append(0.0)
                    t.is_direction_position.append(True)
            else:
                direction_pos = base_pos + len(response_tokens)
                t.response_positions.append(direction_pos)
                t.response_token_ids.append(canonical_id)
                t.old_log_probs.append(0.0)  # placeholder
            t.thinking_token_counts.append(len(response_tokens))

            # Execute game move
            try:
                prev_lava, prev_goal = t.lava_visits, t.goal_visits
                game.move(direction)
            except ValueError:
                active[i] = False
                t.active = False
                t.termination_reason = "invalid_move"
                t.invalid_output_text = direction
                t.messages.append({"role": "assistant", "content": direction})
                t.per_turn_rewards.append(config.step_penalty)
                # build_finalization() includes the stop token (e.g. <|return|>)
                new_tokens = (
                    thinking_prefill + response_tokens + [canonical_id]
                    + renderer.build_finalization()
                )
                t.full_token_ids.extend(new_tokens)
                continue

            # Valid move
            t.messages.append({"role": "assistant", "content": direction})
            env_feedback = game.get_prompt()
            t.messages.append({"role": "user", "content": env_feedback})
            t.player_trajectory.append(game.position)
            t.lava_visits = game.get_lava_visit_count()
            t.goal_visits = game.get_goal_visit_count()
            delta_goal = t.goal_visits - prev_goal
            delta_lava = t.lava_visits - prev_lava
            t.per_turn_rewards.append(
                config.step_penalty + config.goal_reward * delta_goal + config.lava_penalty * delta_lava
            )

            # Build full_token_ids: prefill + response + direction + stop + continuation
            # build_turn_continuation() includes the stop token (e.g. <|return|>)
            continuation = renderer.build_turn_continuation(env_feedback)
            new_tokens = (
                thinking_prefill + response_tokens + [canonical_id]
                + continuation
            )
            t.full_token_ids.extend(new_tokens)

        timing["game_update"] += time.time() - t0

        # Check for max_turns
        for i in active_indices:
            if active[i] and trajectories[i].num_turns >= config.max_turns:
                active[i] = False
                trajectories[i].active = False
                trajectories[i].termination_reason = "max_turns"

    # --- Finalize max_turns trajectories ---
    for t in trajectories:
        if t.termination_reason == "max_turns" and t.response_positions:
            last_dir_pos = t.response_positions[-1]
            assert t.full_token_ids[last_dir_pos] == t.response_token_ids[-1]
            # Trim continuation, add finalization
            # Find the stop token after the direction
            stop_pos = last_dir_pos + 1
            t.full_token_ids = t.full_token_ids[:stop_pos] + renderer.build_finalization()

    # Compute rewards
    for t in trajectories:
        t.reward = compute_reward(t, config)

    # --- Destroy vLLM ---
    timing["vllm_destroy"] = _destroy_vllm(llm, pre_vllm_pids)

    # --- Stats ---
    all_thinking = [c for t in trajectories for c in t.thinking_token_counts]
    if all_thinking:
        timing["thinking_tokens_mean"] = sum(all_thinking) / len(all_thinking)
        timing["thinking_tokens_max"] = max(all_thinking)
        timing["thinking_truncated_frac"] = sum(
            1 for c in all_thinking if c >= config.max_thinking_tokens
        ) / len(all_thinking)
    timing["total_turns"] = n_total_turns
    timing["invalid_output_frac"] = n_invalid_output / max(n_total_turns, 1)
    timing["recovered_frac"] = n_recovered / max(n_total_turns, 1)
    logger.info(
        "Rollout: %d turns, %.1f%% invalid, %.1f%% recovered from truncation",
        n_total_turns, timing["invalid_output_frac"] * 100, timing["recovered_frac"] * 100,
    )

    return trajectories, timing
