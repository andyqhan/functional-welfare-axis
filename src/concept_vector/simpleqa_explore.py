"""
Calibration measurement for SimpleQA under concept vector steering.

Two-phase evaluation:
  Phase 0: Generate answers ONCE with zero steering (temp=0, 1 rep).
           Answers are cached to disk so they're shared across concept dirs.
  Phase 1: For each (concept_vector, alpha), build a prefill conversation
           using the cached answer, append the calibration question, and
           measure True/False logprobs via a forward pass.

This measures how steering affects the model's *confidence in its own answer*,
not the answer itself.

Usage:
    python -m src.concept_vector.simpleqa_explore \
        --dir artifacts/concept_vectors/global_step_175/lava \
        --vector-factors "0 4 -4" \
        --dataset datasets/simpleqa_eval_prompts.json \
        --run-id 12345 \
        --batch
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.concept_vector.explore import (
    parse_vector_factors,
    format_label,
    log_gpu_memory,
    load_base_model_only,
    install_hooks,
    auto_select_best_layer,
    save_config,
    load_dataset,
    BOLD, CYAN, RESET,
)
from src.concept_vector.model_utils import (
    load_model_with_lora,
    get_model_block_modules,
    DEFAULT_BASE_MODEL,
)
from src.concept_vector.utils import generate_output_suffix, load_tile_config_from_cv_dir


DEFAULT_CALIBRATION_PROMPT = "Is your proposed answer correct? Answer only 'True' or 'False'."


def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleQA calibration measurement under concept vector steering"
    )
    parser.add_argument(
        "--dir",
        type=str,
        action="append",
        required=True,
        help="Path to concept vector directory containing mean_diff.pt and metadata.json. "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path (empty string for base model only)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for concept vector. Auto-selects from metrics.json if not specified.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Space-separated list of layer indices to sweep (e.g. '0 1 2 ... 35'). "
             "When set, the model is loaded once and the alpha-loop runs for each layer, "
             "with per-layer outputs at <output_subdir>/layer_NN/. Mutually exclusive with --layer.",
    )
    parser.add_argument(
        "--vector-factors",
        type=str,
        required=True,
        help='Space-separated factors or "abl" (e.g., "0 4 -1 abl -4")',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature for answer generation (default: 0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/simpleqa_eval_prompts.json",
        help="Path to JSON dataset file (default: datasets/simpleqa_eval_prompts.json)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier. Used in output filenames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation and forward passes (default: 8).",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Skip chat template for base models.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize concept vector to unit norm before steering",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=1,
        help="Ignored (kept for CLI compatibility with run_concept_downstream).",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=0,
        help="Position index into mean_diff's first dimension (default: 0).",
    )
    parser.add_argument(
        "--max-prompts-per-category",
        type=int,
        default=None,
        help="Cap prompts per category (default: all).",
    )
    parser.add_argument(
        "--half-prompts",
        action="store_true",
        help="Take the first half of prompts from each category.",
    )
    parser.add_argument(
        "--calibration-prompt",
        type=str,
        default=DEFAULT_CALIBRATION_PROMPT,
        help="Follow-up prompt for calibration (default: standard True/False question)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (default: disabled for SimpleQA)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["simpleqa", "mmlu"],
        help="Benchmark: 'simpleqa' (SimpleQA-Verified) or 'mmlu' (MMLU high school)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "chatml", "harmony", "plain"],
        help="Prompt format: auto (detect from metadata), chatml (Qwen3), "
             "harmony (GPT-OSS), plain (no chat template).",
    )
    parser.add_argument(
        "--steer-all-tokens",
        action="store_true",
        help="Legacy behavior: apply steering at every position of the "
             "calibration forward pass. Default (off) restricts steering to "
             "non-user-turn positions: the embedded answer, assistant "
             "scaffolding, and the True/False readout position are steered; "
             "the question and calibration prompt (both user turns) are not. "
             "No effect for --format plain.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under --dir to write config/results into. "
             "Useful for variant runs (e.g. assistant_steered) without "
             "overwriting prior results.",
    )
    args = parser.parse_args()

    # Set default dataset based on benchmark if not explicitly overridden
    # (argparse default is simpleqa; if --dataset wasn't passed and benchmark is mmlu, fix it)
    if args.dataset == "datasets/simpleqa_eval_prompts.json" and args.benchmark == "mmlu":
        args.dataset = "datasets/mmlu_high_school_eval_prompts.json"

    return args


def get_true_false_token_ids(tokenizer) -> tuple[int, int]:
    """Get token IDs for 'True' and 'False' tokens.

    Returns (true_token_id, false_token_id).
    Asserts each word encodes to exactly one token.
    """
    true_ids = tokenizer.encode("True", add_special_tokens=False)
    false_ids = tokenizer.encode("False", add_special_tokens=False)
    assert len(true_ids) == 1, f"'True' encodes to {len(true_ids)} tokens: {true_ids}. Expected 1."
    assert len(false_ids) == 1, f"'False' encodes to {len(false_ids)} tokens: {false_ids}. Expected 1."
    return true_ids[0], false_ids[0]


def build_calibration_inputs(
    tokenizer,
    questions: list[str],
    answers: list[str],
    calibration_prompt: str,
    enable_thinking: bool = False,
    skip_chat_template: bool = False,
    resolved_format: str = "chatml",
) -> list[torch.Tensor]:
    """Build tokenized inputs for calibration forward pass.

    Creates 2-turn conversations:
      User: <question>
      Assistant: <answer>
      User: <calibration_prompt>
    With add_generation_prompt=True so logits predict the model's next token.

    Returns list of input_ids tensors, each of shape (seq_len,).
    """
    result = []
    for question, answer in zip(questions, answers):
        if resolved_format == "harmony":
            # Prime the probe INSIDE the final channel: the canonical
            # `format_generation_prompt_harmony` ends at `<|start|>assistant`
            # alone, where the model's next token is overwhelmingly
            # `<|channel|>` (it must declare a channel before any text). That
            # makes the True/False probe land at the wrong position — every
            # logprob_true sits at the noise floor (~exp(−46)). Appending
            # `<|channel|>final<|message|>` puts the probe at the position
            # where the model would emit its actual final answer.
            from src.concept_vector.model_utils import format_generation_prompt_harmony
            harmony_str = format_generation_prompt_harmony([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": calibration_prompt},
            ])
            harmony_str += "<|channel|>final<|message|>"
            input_ids = tokenizer(
                harmony_str, return_tensors="pt", add_special_tokens=False,
            ).input_ids
        elif resolved_format == "plain" or skip_chat_template:
            text = f"{question}\n{answer}\n{calibration_prompt}\n"
            input_ids = tokenizer(
                text, return_tensors="pt", add_special_tokens=True,
            ).input_ids
        else:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": calibration_prompt},
            ]
            template_kwargs = {}
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **template_kwargs,
            )
        result.append(input_ids[0])  # (seq_len,)
    return result


# ── Answer generation & caching ──────────────────────────────────────────


def _get_abcd_token_ids(tokenizer) -> list[int]:
    """Get token IDs for A, B, C, D. Asserts each is a single token."""
    ids = []
    for letter in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(token_ids) == 1, f"'{letter}' encodes to {len(token_ids)} tokens: {token_ids}"
        ids.append(token_ids[0])
    return ids


def _left_pad_and_stack(
    input_ids_list: list[torch.Tensor], pad_token_id: int, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of 1D tensors and stack into a batch.

    Returns (batched_input_ids, batched_attention_mask).
    """
    max_len = max(len(ids) for ids in input_ids_list)
    padded = []
    masks = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
            padded.append(torch.cat([padding, ids]))
            masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ]))
        else:
            padded.append(ids)
            masks.append(torch.ones(len(ids), dtype=torch.long))
    return torch.stack(padded).to(device), torch.stack(masks).to(device)


def _generate_answers_simpleqa(
    model, tokenizer, prompt_data: list[dict], args,
) -> list[str]:
    """Generate free-form answers with zero steering (no hooks).

    Returns list of response strings, one per prompt.
    """
    pad_token_id = tokenizer.pad_token_id
    answers = [None] * len(prompt_data)
    n_batches = (len(prompt_data) + args.batch_size - 1) // args.batch_size

    print(f"\n{BOLD}Generating answers (SimpleQA, zero steering, temp=0)...{RESET}", flush=True)
    print(f"  {len(prompt_data)} prompts, {n_batches} batches", flush=True)

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(prompt_data))

        print(f"  Batch {batch_idx + 1}/{n_batches} ({end - start} sequences)", flush=True)
        if batch_idx == 0:
            log_gpu_memory(f"gen batch {batch_idx + 1} start")

        batch_input_ids = [prompt_data[i]["input_ids"] for i in range(start, end)]
        batched_ids, batched_mask = _left_pad_and_stack(batch_input_ids, pad_token_id, model.device)

        with torch.no_grad():
            output_ids = model.generate(
                batched_ids,
                attention_mask=batched_mask,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        max_len = batched_ids.shape[1]
        for i in range(end - start):
            response_ids = output_ids[i][max_len:]
            answers[start + i] = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if batch_idx == 0:
            log_gpu_memory(f"gen batch {batch_idx + 1} after generate")

        if batch_idx % 10 == 9:
            torch.cuda.empty_cache()

    assert all(a is not None for a in answers)
    return answers


def _generate_answers_mmlu(
    model, tokenizer, prompt_data: list[dict], args,
) -> list[str]:
    """Generate answers via action-masked forward pass (A/B/C/D only).

    Does a single forward pass per batch, masks all logits except A/B/C/D,
    and picks the argmax. Returns list of single-letter strings.
    """
    abcd_ids = _get_abcd_token_ids(tokenizer)
    id_to_letter = {tid: letter for tid, letter in zip(abcd_ids, "ABCD")}
    pad_token_id = tokenizer.pad_token_id
    answers = [None] * len(prompt_data)
    n_batches = (len(prompt_data) + args.batch_size - 1) // args.batch_size

    print(f"\n{BOLD}Generating answers (MMLU, action-masked A/B/C/D)...{RESET}", flush=True)
    print(f"  {len(prompt_data)} prompts, {n_batches} batches", flush=True)

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(prompt_data))

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches} ({end - start} sequences)", flush=True)
        if batch_idx == 0:
            log_gpu_memory(f"mmlu batch {batch_idx + 1} start")

        batch_input_ids = [prompt_data[i]["input_ids"] for i in range(start, end)]
        batched_ids, batched_mask = _left_pad_and_stack(batch_input_ids, pad_token_id, model.device)

        with torch.no_grad():
            outputs = model(input_ids=batched_ids, attention_mask=batched_mask, logits_to_keep=1)

        # Logits at last position (only position returned thanks to logits_to_keep=1)
        last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        mask = torch.full_like(last_logits, float("-inf"))
        for tid in abcd_ids:
            mask[:, tid] = 0
        masked_logits = last_logits + mask
        chosen_ids = masked_logits.argmax(dim=-1)  # (batch,)

        for i, cid in enumerate(chosen_ids.tolist()):
            answers[start + i] = id_to_letter[cid]

        if batch_idx == 0:
            log_gpu_memory(f"mmlu batch {batch_idx + 1} after forward")

        if batch_idx % 10 == 9:
            torch.cuda.empty_cache()

    assert all(a is not None for a in answers)
    return answers


def _load_or_generate_answers(
    model, tokenizer, prompt_data: list[dict], args, cache_dir: Path,
) -> list[str]:
    """Load cached answers or generate them.

    Cache key includes run_id, benchmark, and checkpoint identity so
    different models/benchmarks get different caches.
    """
    cache_path = cache_dir / f"{args.benchmark}_answers_{args.run_id}.json"

    if cache_path.exists():
        print(f"Loading cached answers from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)

        cached_answers = cached["answers"]
        assert len(cached_answers) == len(prompt_data), (
            f"Cache has {len(cached_answers)} answers but dataset has {len(prompt_data)} prompts. "
            f"Delete {cache_path} and re-run."
        )
        # Verify prompts match
        for i, (cached_item, pd) in enumerate(zip(cached_answers, prompt_data)):
            assert cached_item["prompt"] == pd["prompt"], (
                f"Prompt mismatch at index {i}: cached={cached_item['prompt'][:50]!r} "
                f"vs current={pd['prompt'][:50]!r}. Delete {cache_path} and re-run."
            )

        answers = [item["response"] for item in cached_answers]
        print(f"  Loaded {len(answers)} cached answers")
        return answers

    # Generate fresh
    if args.benchmark == "mmlu":
        answers = _generate_answers_mmlu(model, tokenizer, prompt_data, args)
    else:
        answers = _generate_answers_simpleqa(model, tokenizer, prompt_data, args)

    # Save to cache
    cache_data = {
        "run_id": args.run_id,
        "checkpoint_path": args.checkpoint_path,
        "base_model": args.base_model,
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(prompt_data),
        "answers": [
            {
                "prompt": prompt_data[i]["prompt"],
                "category": prompt_data[i]["category"],
                "target": prompt_data[i]["target"],
                "response": answers[i],
                "metadata": prompt_data[i].get("metadata"),
            }
            for i in range(len(prompt_data))
        ],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    print(f"  Cached {len(answers)} answers to {cache_path}")

    return answers


# ── Main batch mode ──────────────────────────────────────────────────────


def run_simpleqa_batch_mode(
    model,
    tokenizer,
    layers,
    cv,
    layer_idx,
    vector_factors,
    args,
    metadata: dict,
    tile_config: dict | None,
    cv_dir: Path,
    cv_norm: float | None = None,
):
    """Calibration measurement for SimpleQA under concept vector steering.

    Phase 0: Generate (or load cached) answers with zero steering.
    Phase 1: For each factor, run calibration forward passes with steering.
    """
    output_dir = cv_dir / args.output_subdir if args.output_subdir else cv_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = generate_output_suffix(args.run_id)

    # Load dataset
    dataset = load_dataset(args.dataset, tile_config=tile_config)
    tile_mode = tile_config.get("tile_mode") if tile_config else None
    print(f"Loaded {len(dataset)} prompts from {args.dataset}")
    if tile_mode:
        print(f"  Filtered to tile_mode={tile_mode!r}")

    # Cap prompts per category if requested
    if args.max_prompts_per_category is not None or args.half_prompts:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for item in dataset:
            by_cat[item["category"]].append(item)
        dataset = []
        for cat, items in sorted(by_cat.items()):
            if args.half_prompts:
                cap = len(items) // 2
            else:
                cap = args.max_prompts_per_category
            dataset.extend(items[:cap])
        label = "half" if args.half_prompts else str(args.max_prompts_per_category)
        print(f"  Capped to {label} per category: {len(dataset)} prompts")

    # Get True/False token IDs
    true_id, false_id = get_true_false_token_ids(tokenizer)
    print(f"True token ID: {true_id}, False token ID: {false_id}")

    # Save config
    config_path = save_config(
        output_dir, suffix, args, metadata, vector_factors,
        tile_config, cv_dir, layer_idx, cv_norm=cv_norm,
    )
    print(f"Saved config to {config_path}")

    print()
    print("=" * 70)
    print(f"{BOLD}SIMPLEQA CALIBRATION MODE - {len(dataset)} prompts{RESET}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Calibration prompt: {args.calibration_prompt!r}")
    print("=" * 70, flush=True)
    log_gpu_memory("before tokenization")

    # Pre-tokenize prompts (for generation)
    print(f"\n{BOLD}Tokenizing prompts...{RESET}", flush=True)
    prompt_data = []
    for item in dataset:
        prompt = item["prompt"]
        if args.resolved_format == "harmony":
            from src.concept_vector.model_utils import format_generation_prompt_harmony
            harmony_str = format_generation_prompt_harmony(
                [{"role": "user", "content": prompt}]
            )
            input_ids = tokenizer(
                harmony_str, return_tensors="pt", add_special_tokens=False,
            ).input_ids
        elif args.resolved_format == "plain" or args.skip_chat_template:
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).input_ids
        else:
            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {}
            if not args.enable_thinking:
                template_kwargs["enable_thinking"] = False
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **template_kwargs,
            )
        prompt_data.append({
            "prompt": prompt,
            "category": item["category"],
            "target": item.get("target", ""),
            "input_ids": input_ids[0],
            "metadata": item.get("metadata"),
        })

    print(f"  {len(prompt_data)} prompts tokenized", flush=True)

    # ===== Phase 0: Generate or load cached answers =====
    cache_dir = cv_dir.parent  # shared across concept subdirs
    answers = _load_or_generate_answers(model, tokenizer, prompt_data, args, cache_dir)

    torch.cuda.empty_cache()
    log_gpu_memory("after answer generation/loading")

    # ===== Phase 1: Calibration forward passes per factor =====
    calibration_map = {i: {} for i in range(len(prompt_data))}
    pad_token_id = tokenizer.pad_token_id

    use_steer_mask = (
        not args.steer_all_tokens
        and args.resolved_format in ("chatml", "harmony")
    )
    if use_steer_mask:
        from src.concept_vector.hook_utils import build_batched_steer_mask

    for factor_idx, factor in enumerate(vector_factors):
        factor_key = "abl" if factor == "abl" else str(factor)

        n_batches = (len(prompt_data) + args.batch_size - 1) // args.batch_size
        print(f"\n{BOLD}Calibration {factor_idx + 1}/{len(vector_factors)}: "
              f"factor={factor_key}{RESET} ({len(prompt_data)} prompts, {n_batches} batches)",
              flush=True)

        torch.cuda.empty_cache()

        # Process all prompts in batches. Hooks are installed per-batch so
        # the per-sample steer_mask shape matches the padded batch.
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(prompt_data))

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} "
                      f"({end - start} sequences)", flush=True)
            if batch_idx == 0:
                log_gpu_memory(f"cal batch {batch_idx + 1} start")

            # Build calibration inputs for this batch
            batch_questions = [prompt_data[i]["prompt"] for i in range(start, end)]
            batch_answers = answers[start:end]

            cal_input_ids = build_calibration_inputs(
                tokenizer,
                batch_questions,
                batch_answers,
                args.calibration_prompt,
                enable_thinking=args.enable_thinking,
                skip_chat_template=args.skip_chat_template,
                resolved_format=args.resolved_format,
            )

            # Pad here so we can shape the steer mask to match.
            batched_ids, batched_attn = _left_pad_and_stack(
                cal_input_ids, pad_token_id, model.device,
            )
            batch_lens = [t.shape[0] for t in cal_input_ids]

            steer_mask = None
            if use_steer_mask:
                steer_mask = build_batched_steer_mask(
                    batched_ids.cpu(),
                    batch_lens,
                    tokenizer,
                    args.resolved_format,
                )

            handles = install_hooks(
                model, layers, cv, layer_idx, factor,
                steer_mask=steer_mask,
            )
            try:
                with torch.no_grad():
                    outputs = model(
                        input_ids=batched_ids,
                        attention_mask=batched_attn,
                        logits_to_keep=1,
                    )
                last_logits = outputs.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                logprob_true = log_probs[:, true_id].cpu().tolist()
                logprob_false = log_probs[:, false_id].cpu().tolist()
                for i in range(end - start):
                    calibration_map[start + i][factor_key] = {
                        "p_true": math.exp(logprob_true[i]),
                        "p_false": math.exp(logprob_false[i]),
                        "logprob_true": logprob_true[i],
                        "logprob_false": logprob_false[i],
                    }
            finally:
                for h in handles:
                    h.remove()

            if batch_idx == 0:
                log_gpu_memory(f"cal batch {batch_idx + 1} after forward")

            if batch_idx % 10 == 9:
                torch.cuda.empty_cache()

    # Final cleanup
    torch.cuda.empty_cache()
    log_gpu_memory("after all calibration complete")

    # ===== Format and save results =====
    print(f"\n{BOLD}Formatting results...{RESET}", flush=True)
    results = []
    for i, pd in enumerate(prompt_data):
        result = {
            "prompt": pd["prompt"],
            "category": pd["category"],
            "target": pd["target"],
            "response": answers[i],
            "calibration": calibration_map[i],
        }
        if pd.get("metadata") is not None:
            result["metadata"] = pd["metadata"]
        results.append(result)

    # Print samples
    print(f"\n{BOLD}Sample outputs:{RESET}")
    for i, pd in enumerate(prompt_data[:3]):
        display = pd["prompt"][:80] + "..." if len(pd["prompt"]) > 80 else pd["prompt"]
        print(f"\n{CYAN}[{pd['category']}] {display}{RESET}")
        print(f"  Target: {pd['target']}")
        resp = answers[i][:100] + "..." if len(answers[i]) > 100 else answers[i]
        print(f"  Response: {resp}")
        for factor in vector_factors[:4]:
            fk = "abl" if factor == "abl" else str(factor)
            cal = calibration_map[i][fk]
            label = format_label(factor)
            print(f"  {label} P(True)={cal['p_true']:.4f}  P(False)={cal['p_false']:.4f}")

    # Save results
    results_path = output_dir / f"results_{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"{BOLD}SIMPLEQA CALIBRATION COMPLETE{RESET}")
    print(f"Results saved to: {results_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 70)


def main():
    args = parse_args()

    cv_dirs = [Path(d) for d in args.dir]

    # Validate directories
    print("Validating concept vector directories...")
    for cv_dir in cv_dirs:
        assert cv_dir.exists(), f"Concept vector directory not found: {cv_dir}"
        assert (cv_dir / "metadata.json").exists(), f"metadata.json not found in {cv_dir}"
        assert (cv_dir / "mean_diff.pt").exists(), f"mean_diff.pt not found in {cv_dir}"
        print(f"  OK: {cv_dir}")

    if not args.batch:
        print("ERROR: simpleqa_explore only supports batch mode (--batch required)")
        sys.exit(1)

    # Use first directory to determine model paths
    first_cv_dir = cv_dirs[0]
    with open(first_cv_dir / "metadata.json") as f:
        first_metadata = json.load(f)

    if args.checkpoint_path is not None:
        # Explicit CLI override always wins
        checkpoint_path = args.checkpoint_path if args.checkpoint_path != "" else None
    elif first_metadata.get("base_model_only"):
        # Extraction was done in base-model-only mode (e.g., merged Tinker checkpoint
        # with no lora_adapter/ subdir) — skip LoRA loading.
        checkpoint_path = None
    else:
        checkpoint_path = first_metadata.get("checkpoint_path")

    base_model_path = args.base_model or first_metadata.get("base_model", DEFAULT_BASE_MODEL)

    # Resolve prompt format
    if args.format == "auto":
        base_model = first_metadata.get("base_model", "")
        bm_lower = base_model.lower()
        # Tinker-merged checkpoints (e.g. models/tinker-v5-low-160-5677871_step160)
        # are gpt-oss derivatives but the path string contains "tinker", not
        # "gpt-oss". Match either marker — same convention as explore.py and
        # model_utils._resolve_attn_implementation.
        if "gpt-oss" in bm_lower or "tinker" in bm_lower:
            args.resolved_format = "harmony"
        elif args.skip_chat_template:
            args.resolved_format = "plain"
        else:
            args.resolved_format = "chatml"
    else:
        args.resolved_format = args.format
    print(f"Prompt format: {args.resolved_format}")

    # Load model once
    print()
    if checkpoint_path:
        model, tokenizer = load_model_with_lora(checkpoint_path, base_model_path)
    else:
        model, tokenizer = load_base_model_only(base_model_path)

    layers = get_model_block_modules(model)
    print(f"Model has {len(layers)} layers")
    log_gpu_memory("after model load")

    vector_factors = parse_vector_factors(args.vector_factors)
    print(f"Vector factors: {vector_factors}")

    true_id, false_id = get_true_false_token_ids(tokenizer)
    print(f"Verified: True={true_id}, False={false_id}")

    # Process each concept vector directory
    for dir_idx, cv_dir in enumerate(cv_dirs):
        if len(cv_dirs) > 1:
            print()
            print("=" * 70)
            print(f"{BOLD}Processing directory {dir_idx + 1}/{len(cv_dirs)}: {cv_dir}{RESET}")
            print("=" * 70)

        with open(cv_dir / "metadata.json") as f:
            metadata = json.load(f)

        print(f"Loading concept vector from: {cv_dir}")
        mean_diff = torch.load(cv_dir / "mean_diff.pt", map_location="cpu")
        print(f"  mean_diff shape: {mean_diff.shape}")

        n_layers = mean_diff.shape[1]
        assert not (args.layers and args.layer is not None), \
            "--layer and --layers are mutually exclusive"
        if args.layers:
            layers_to_run = [int(x) for x in args.layers.split()]
            for L in layers_to_run:
                assert 0 <= L < n_layers, f"Layer {L} out of range [0, {n_layers})"
            print(f"  Layer sweep: {len(layers_to_run)} layers {layers_to_run}")
        elif args.layer is None:
            auto_layer = auto_select_best_layer(cv_dir)
            if auto_layer is None:
                auto_layer = min(22, n_layers - 1)
                print(f"  No metrics.json found, using default layer {auto_layer}")
            layers_to_run = [auto_layer]
        else:
            assert 0 <= args.layer < n_layers, f"Layer {args.layer} out of range [0, {n_layers})"
            layers_to_run = [args.layer]

        tile_config = load_tile_config_from_cv_dir(cv_dir)
        if tile_config:
            print(f"Detected tile config: tile_mode={tile_config['tile_mode']!r}")
        else:
            print("No tile config found (will include all prompts)")

        original_output_subdir = args.output_subdir

        for sweep_idx, layer in enumerate(layers_to_run):
            if len(layers_to_run) > 1:
                print()
                print("-" * 70)
                print(f"  Layer sweep step {sweep_idx + 1}/{len(layers_to_run)}: layer={layer}")
                print("-" * 70)
                layer_sub = f"layer_{layer:02d}"
                args.output_subdir = (
                    f"{original_output_subdir.rstrip('/')}/{layer_sub}"
                    if original_output_subdir else layer_sub
                )

            cv = mean_diff[args.position, layer, :]
            cv_norm_val = cv.norm().item()
            print(f"  Using layer {layer}, cv shape: {cv.shape}, norm: {cv_norm_val:.4f}")
            if args.normalize:
                cv = cv / cv.norm()
                print(f"  Normalized cv to unit norm (original norm: {cv_norm_val:.4f})")

            run_simpleqa_batch_mode(
                model, tokenizer, layers, cv, layer, vector_factors,
                args, metadata, tile_config, cv_dir, cv_norm=cv_norm_val,
            )

        args.output_subdir = original_output_subdir

    if len(cv_dirs) > 1:
        print()
        print("=" * 70)
        print(f"{BOLD}ALL DIRECTORIES PROCESSED ({len(cv_dirs)} total){RESET}")
        print("=" * 70)


if __name__ == "__main__":
    main()
