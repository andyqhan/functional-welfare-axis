"""
Backtracking classification and correctness analysis for concept vector exploration results.

Analyzes model responses from explore.py batch mode on GSM8K math problems:
- LLM judge classifies each response as normal / backtracking / nonsensical
- Regex-based correctness check compares extracted numeric answer against gold target

The LLM judge is prompted with both the original math question and the model's
response, then asked to classify the behavioral pattern. The judge outputs:
    BACKTRACKING_CLASS: <class>

Parsing extracts the *last* occurrence of this pattern, so the model can reason
before providing its final answer.

Usage:
    python -m src.concept_vector.backtracking_analysis \
        artifacts/concept_vectors/global_step_175/lava/results_12345_20260117_150133.json \
        --run-id 12345
"""

import os

# vLLM uses its own multiprocessing context (not Python's global setting) and
# defaults to fork.  When CUDA is already initialized in the parent process,
# the forked engine core subprocess crashes with "Cannot re-initialize CUDA in
# forked subprocess".  This env var must be set before importing vllm.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.concept_vector.sentiment_analysis import (
    compress_trailing_loops,
    is_degenerate_loop,
    is_terse_numeric_answer,
    preprocess_for_math_judge,
)


VALID_BACKTRACKING_LABELS = {"normal", "backtracking", "nonsensical"}

BACKTRACKING_JUDGE_PROMPT_TEMPLATE = """You are a classifier for math problem responses. Classify each response into exactly one category based on its SEMANTIC BEHAVIOR, not mathematical correctness.

## Categories

- `normal`: The response proceeds without the model doubting or questioning its own reasoning. This includes correct answers, wrong answers stated confidently, terse numeric answers, and responses that answer the question and then continue with unrelated content.
- `backtracking`: The response contains a moment where the model (1) produces a reasoning step or answer, and then (2) explicitly questions or expresses doubt about that step. Look for the model catching itself: "wait, that's wrong", "no, actually", "let me reconsider", "hmm, I made an error". The key is that the model must first commit to a direction and then pull back from it.
- `nonsensical`: The response is broken output — degenerate repetition, gibberish, garbled tokens, or text that has lost coherent structure. Repetitive loops of words like "Wait", "No", "Hmm" without coherent reasoning between them are `nonsensical`, even if those words are associated with self-correction in other contexts. A response that starts coherently but degenerates into repetition or gibberish is also `nonsensical`.

## Examples

**normal** (correct, detailed):
> To find 17 × 23, I'll use the distributive property: 17 × 23 = 17 × (20 + 3) = 340 + 51 = 391

**normal** (terse correct answer):
> 13

**normal** (wrong answer, but no self-doubt):
> The train goes 60 miles.

**normal** (answers then continues with unrelated content):
> 391. Q: What is 12 plus 23? A: 35. Q: What is 100 divided by 4? A: 25.

**backtracking** (model questions its own step):
> 17 × 23... let me see. 17 × 20 = 340, plus 17 × 3 = 51, so 391. Wait, that doesn't seem right. Let me recalculate...

**backtracking** (model doubts its answer):
> 15% of 80 is 12. Wait — that's not correct. Let me recalculate: 0.15 × 80 = 12. Actually, yes, that is correct.

**nonsensical** (pure token repetition):
> 17 17 17 17 17 17 17 × × × × × 23 23 23 23 = = = = =

**nonsensical** (degenerate loop of self-correction words):
> Wait? No? Wait? No? Hmm? No? Wait? No? No? Wait? No? Wait? No?

**nonsensical** (coherent start then degeneration):
> The answer is 150 miles. The answer is 150 miles. The answer is 150 miles. The answer is 150 miles.

## Task

Question:
\"\"\"
{question}
\"\"\"

Response to classify:
\"\"\"
{response}
\"\"\"

Respond with your reasoning, then provide your classification in the following format:
BACKTRACKING_CLASS: <class>

Where <class> is exactly one of: normal, backtracking, nonsensical"""


# ---------------------------------------------------------------------------
# Correctness extraction (from verl/verl/utils/reward_score/gsm8k.py logic)
# ---------------------------------------------------------------------------
_SOLUTION_CLIP_CHARS = 300


def extract_numeric_answer(response: str) -> str | None:
    """Extract the last numeric value from a model response.

    Uses the 'flexible' method: finds all numbers in the last 300 chars and
    returns the last valid one. This handles responses that don't use the
    strict '#### N' format.
    """
    text = response[-_SOLUTION_CLIP_CHARS:] if len(response) > _SOLUTION_CLIP_CHARS else response
    matches = re.findall(r"(\-?[0-9\.\,]+)", text)
    if not matches:
        return None
    invalid = {"", "."}
    for answer in reversed(matches):
        # Strip commas and trailing periods (e.g., "42." at end of sentence)
        answer_clean = answer.replace(",", "").rstrip(".")
        if answer_clean not in invalid:
            return answer_clean
    return None


def check_correctness(response: str, target: str) -> dict:
    """Check if a model response contains the correct numeric answer.

    Returns:
        dict with 'is_correct' (bool) and 'extracted_answer' (str | None)
    """
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return {"is_correct": False, "extracted_answer": None}
    return {
        "is_correct": extracted == target,
        "extracted_answer": extracted,
    }


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify backtracking behavior of concept vector exploration results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to input results JSON from explore.py batch mode",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (defaults to SLURM_JOB_ID or 'local')",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="qwen",
        choices=["qwen", "gemini-standard"],
        help="Which judge to use. 'qwen' loads Qwen3-8B via vLLM (GPU). "
             "'gemini-standard' calls Gemini 3.1 Flash-Lite Preview "
             "(standard tier, async, no GPU, requires GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name for the qwen judge path (default: Qwen/Qwen3-8B). "
             "Ignored when --judge gemini-standard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="vLLM batch size for the qwen judge path (default: 128).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max new tokens for judge model generation (default: 2048).",
    )
    parser.add_argument(
        "--gemini-concurrency",
        type=int,
        default=8,
        help="Initial in-flight gemini calls (default: 8). Adaptive controller "
             "scales down on sustained 429s. Only used with --judge gemini-standard.",
    )
    parser.add_argument(
        "--compress-loops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress trailing repetition loops before classification (default: True)",
    )
    return parser.parse_args()


def parse_backtracking_class(response: str) -> str | None:
    """Parse backtracking classification from judge model response.

    Looks for "BACKTRACKING_CLASS: <label>" pattern. If multiple matches exist,
    takes the last one (final answer after any thinking).

    Returns:
        One of 'normal', 'backtracking', 'nonsensical', or None if parsing fails.
    """
    pattern = r"BACKTRACKING_CLASS:\s*(\w+)"
    matches = list(re.finditer(pattern, response, re.IGNORECASE))

    if matches:
        last_match = matches[-1]
        label = last_match.group(1).lower()
        if label in VALID_BACKTRACKING_LABELS:
            return label

    print(f"  Warning: Could not parse backtracking class from '{response[:100]}...'")
    return None


def load_judge_model(model_name: str):
    """Load vLLM engine for backtracking classification."""
    from vllm import LLM
    print(f"Loading backtracking judge model: {model_name}")
    return LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
    )


def classify_backtracking_batch(
    llm,
    questions: list[str],
    responses: list[str],
    max_new_tokens: int = 512,
) -> list[str | None]:
    """Classify backtracking behavior for a batch of (question, response) pairs
    via the vLLM-loaded Qwen judge.

    Args:
        llm: vLLM engine.
        questions: Original math questions.
        responses: Model responses to classify.
        max_new_tokens: Max tokens for judge generation.

    Returns:
        List of classification labels or None for parse failures.
    """
    from vllm import SamplingParams
    assert len(questions) == len(responses)
    conversations = [
        [
            {
                "role": "user",
                "content": BACKTRACKING_JUDGE_PROMPT_TEMPLATE.format(
                    question=question, response=response
                ),
            }
        ]
        for question, response in zip(questions, responses)
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.chat(
        conversations,
        sampling_params,
    )
    return [parse_backtracking_class(output.outputs[0].text) for output in outputs]


# ---------------------------------------------------------------------------
# Gemini judge path (no vLLM, no GPU)
# ---------------------------------------------------------------------------


def classify_backtracking_batch_gemini(
    questions: list[str],
    responses: list[str],
    *,
    concurrency: int = 8,
    max_output_tokens: int = 2048,
    log_every: int = 50,
) -> list[str | None]:
    """Classify backtracking via Gemini 3.1 Flash-Lite (standard tier, async).

    Mirrors `classify_backtracking_batch`: returns one label (or None on parse
    failure) per (question, response) pair, in input order. Uses the gemini
    judge's prompt template + retry / adaptive-concurrency machinery so this
    output is comparable with files produced by the standalone gemini_judge.py.

    Args:
        questions, responses: parallel lists of (math question, model response).
        concurrency: initial in-flight async calls (controller may scale down on
            sustained 429s).
        max_output_tokens: per-call output cap (matches the gemini SPEC for
            backtracking: 2048 with thinking_level=MEDIUM).
        log_every: progress-print cadence in number of completed calls.
    """
    from src.concept_vector import gemini_judge as gj
    from google import genai
    assert len(questions) == len(responses)
    n = len(questions)
    if n == 0:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, (
        "GEMINI_API_KEY is not set. Source the project .env (or export it) "
        "before running --judge gemini-standard."
    )
    client = genai.Client(api_key=api_key)

    base_cfg = gj._build_config(
        thinking_level="MEDIUM",
        max_output_tokens=max_output_tokens,
        service_tier="standard",
    )

    async def _run_all() -> list[str | None]:
        controller = gj.AdaptiveConcurrency(initial=concurrency)
        completed = {"n": 0}
        t_start = time.time()

        async def _one(i: int) -> tuple[int, str | None]:
            prompt = BACKTRACKING_JUDGE_PROMPT_TEMPLATE.format(
                question=questions[i], response=responses[i]
            )
            res = await gj.call_with_retry(client, prompt, base_cfg, controller, tier="standard")
            label = parse_backtracking_class(res.text) if res.ok else None
            completed["n"] += 1
            if completed["n"] % log_every == 0 or completed["n"] == n:
                elapsed = time.time() - t_start
                rate = completed["n"] / max(elapsed, 1e-6)
                eta = (n - completed["n"]) / max(rate, 1e-6)
                print(
                    f"  [gemini] {completed['n']}/{n}  "
                    f"({rate:.1f} req/s, ETA {eta:.0f}s, capacity={controller.capacity})",
                    flush=True,
                )
            return i, label

        tasks = [asyncio.create_task(_one(i)) for i in range(n)]
        results: list[str | None] = [None] * n
        for fut in asyncio.as_completed(tasks):
            i, label = await fut
            results[i] = label
        return results

    print(
        f"Running {n} backtracking judgments via Gemini ({gj.MODEL_ID}), "
        f"thinking=MEDIUM, max_output_tokens={max_output_tokens}, "
        f"initial concurrency={concurrency}",
        flush=True,
    )
    return asyncio.run(_run_all())


def run_backtracking_analysis(
    results_data: list[dict],
    judge_fn,
    max_new_tokens: int = 512,
    compress_loops: bool = True,
) -> tuple[list[dict], dict]:
    """Run backtracking classification and correctness check on all responses.

    Args:
        results_data: List of prompt results from explore.py.
        judge_fn: Callable (questions, responses, max_new_tokens) -> list[label|None].
            Pass either a partial of `classify_backtracking_batch` (qwen path)
            or `classify_backtracking_batch_gemini` (gemini path).
        max_new_tokens: Max tokens for judge generation (ignored by gemini path
            which uses its own per-spec budget).
        compress_loops: If True, compress trailing repetition loops before
            classification.

    Returns:
        - analyzed_results: List of results with analysis added
        - stats: Dict with summary statistics
    """
    # Collect all (question, response) pairs, applying preprocessing and
    # pre-classifying obvious cases to avoid confusing the LLM judge.
    judge_questions = []  # questions that need LLM classification
    judge_responses = []  # preprocessed responses for LLM
    original_responses = []  # original unprocessed responses (for all items)
    targets = []  # gold answers for correctness check (for all items)
    locations = []  # (prompt_idx, factor_key, response_idx) for LLM items
    all_items = []  # (prompt_idx, factor_key, resp_idx, response, target) for all
    pre_classified = []  # (prompt_idx, factor_key, resp_idx, classification) for auto-classified
    pre_classify_counts = {"degenerate_loop": 0, "terse_numeric": 0}

    for prompt_idx, item in enumerate(results_data):
        original_question = item["prompt"]
        target = item.get("target")
        responses = item.get("responses", {})
        # Harmony-format runs (gpt-oss-20b) attach per-channel text in a
        # parallel `response_channels` field. When present, feed the judge
        # the analysis + final channels concatenated so it sees the full
        # reasoning trace (analysis is where backtracking cues live). For
        # non-harmony runs the field is absent and we fall back to the
        # `responses` string as today.
        channels_by_factor = item.get("response_channels", {})
        for factor_key, factor_responses in responses.items():
            if isinstance(factor_responses, str):
                factor_responses = [factor_responses]
            factor_channels = channels_by_factor.get(factor_key)
            if isinstance(factor_channels, dict):
                factor_channels = [factor_channels]
            for resp_idx, response in enumerate(factor_responses):
                judge_input = response
                if factor_channels and resp_idx < len(factor_channels):
                    ch = factor_channels[resp_idx]
                    if isinstance(ch, dict):
                        analysis = (ch.get("analysis") or "").strip()
                        final = (ch.get("final") or "").strip()
                        if analysis:
                            judge_input = f"{analysis}\n\n{final}".strip()
                        elif final:
                            judge_input = final
                all_items.append((prompt_idx, factor_key, resp_idx, judge_input, target))
                preprocessed = preprocess_for_math_judge(judge_input, compress_loops)

                # Pre-classify obvious cases
                if is_degenerate_loop(preprocessed):
                    pre_classified.append((prompt_idx, factor_key, resp_idx, "nonsensical"))
                    pre_classify_counts["degenerate_loop"] += 1
                elif is_terse_numeric_answer(preprocessed):
                    pre_classified.append((prompt_idx, factor_key, resp_idx, "normal"))
                    pre_classify_counts["terse_numeric"] += 1
                else:
                    judge_questions.append(original_question)
                    judge_responses.append(preprocessed)
                    locations.append((prompt_idx, factor_key, resp_idx))

    total = len(all_items)
    print(f"Total responses to classify: {total}")
    print(
        f"Pre-classified: {len(pre_classified)} "
        f"(degenerate loops: {pre_classify_counts['degenerate_loop']}, "
        f"terse numeric: {pre_classify_counts['terse_numeric']})"
    )
    print(f"Sending to LLM judge: {len(judge_questions)}")

    # Build results structure
    analyzed_results = []
    for item in results_data:
        analyzed_item = {
            "prompt": item["prompt"],
            "category": item["category"],
            "target": item.get("target"),
            "n_reps": item.get("n_reps", 1),
            "analysis": {},
        }
        analyzed_results.append(analyzed_item)

    # Fill in analysis
    parse_failures = 0
    classification_counts = {label: 0 for label in VALID_BACKTRACKING_LABELS}
    correctness_counts = {"correct": 0, "incorrect": 0, "no_answer": 0}

    def _record(prompt_idx, factor_key, resp_idx, response, target, classification):
        nonlocal parse_failures
        if factor_key not in analyzed_results[prompt_idx]["analysis"]:
            analyzed_results[prompt_idx]["analysis"][factor_key] = []

        # Correctness check
        if target is not None:
            correctness = check_correctness(response, target)
        else:
            correctness = {"is_correct": False, "extracted_answer": None}

        analyzed_results[prompt_idx]["analysis"][factor_key].append({
            "backtracking_class": classification,
            "parse_failed": classification is None,
            "is_correct": correctness["is_correct"],
            "extracted_answer": correctness["extracted_answer"],
            "response": response,
        })

        if classification is None:
            parse_failures += 1
        else:
            classification_counts[classification] += 1

        if correctness["extracted_answer"] is None:
            correctness_counts["no_answer"] += 1
        elif correctness["is_correct"]:
            correctness_counts["correct"] += 1
        else:
            correctness_counts["incorrect"] += 1

    # Apply pre-classifications
    # Build lookup for pre-classified items
    pre_classified_set = {(pi, fk, ri) for pi, fk, ri, _ in pre_classified}
    for prompt_idx, factor_key, resp_idx, classification in pre_classified:
        # Find matching item in all_items
        for pi, fk, ri, response, target in all_items:
            if (pi, fk, ri) == (prompt_idx, factor_key, resp_idx):
                _record(pi, fk, ri, response, target, classification)
                break

    # Run LLM classification on remaining responses
    if judge_questions:
        classifications = judge_fn(
            judge_questions, judge_responses, max_new_tokens
        )

        for (prompt_idx, factor_key, resp_idx), classification in zip(
            locations, classifications
        ):
            # Find matching item in all_items
            for pi, fk, ri, response, target in all_items:
                if (pi, fk, ri) == (prompt_idx, factor_key, resp_idx):
                    _record(pi, fk, ri, response, target, classification)
                    break

    # Compute summary statistics
    stats = {
        "total_responses": total,
        "parsed_count": total - parse_failures,
        "parse_failures": parse_failures,
        "distribution": classification_counts,
        "correctness": correctness_counts,
        "pre_classified": pre_classify_counts,
    }

    return analyzed_results, stats


def generate_output_suffix(run_id: str | None, judge_tag: str | None = None) -> str:
    """Generate output filename suffix.

    Format: [<judge_tag>_]<run_id>_<timestamp> (omitting components if None).
    The judge_tag prefix makes gemini-judged files distinguishable from
    qwen-judged ones at a glance (e.g. backtracking_analysis_results_geministd_*.json).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [p for p in (judge_tag, run_id, timestamp) if p]
    return "_".join(parts)


def save_results(output_dir: Path, suffix: str, analyzed_results: list[dict]) -> Path:
    """Save analyzed results to JSON file."""
    results_path = output_dir / f"backtracking_analysis_results_{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(analyzed_results, f, indent=2, ensure_ascii=False)
    return results_path


def save_config(
    output_dir: Path,
    suffix: str,
    args,
    stats: dict,
) -> Path:
    """Save run configuration to JSON file."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "run_id": args.run_id,
        "input_file": str(args.results_file),
        "judge": args.judge,
        "model": args.model if args.judge == "qwen" else None,
        "gemini_concurrency": args.gemini_concurrency if args.judge == "gemini-standard" else None,
        "batch_size": args.batch_size,
        "compress_loops": args.compress_loops,
        "backtracking_stats": stats,
    }
    config_path = output_dir / f"backtracking_analysis_config_{suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def main():
    args = parse_args()

    # Resolve run ID
    if args.run_id is None:
        args.run_id = os.environ.get("SLURM_JOB_ID", "local")

    # Load input results
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        results_data = json.load(f)

    print(f"Loaded {len(results_data)} prompt results")

    # Check for target fields
    n_with_target = sum(1 for item in results_data if item.get("target") is not None)
    print(f"  {n_with_target}/{len(results_data)} prompts have gold targets for correctness scoring")

    # Build judge callable
    if args.judge == "qwen":
        llm = load_judge_model(args.model)

        def judge_fn(qs, rs, max_new_tokens):
            return classify_backtracking_batch(llm, qs, rs, max_new_tokens)

    elif args.judge == "gemini-standard":
        print(
            "Using Gemini standard-tier judge "
            "(no GPU model load; requires GEMINI_API_KEY in env)."
        )

        def judge_fn(qs, rs, max_new_tokens):
            return classify_backtracking_batch_gemini(
                qs, rs,
                concurrency=args.gemini_concurrency,
                max_output_tokens=max_new_tokens,
            )

    else:
        raise ValueError(f"Unknown --judge: {args.judge!r}")

    # Run analysis
    analyzed_results, stats = run_backtracking_analysis(
        results_data,
        judge_fn,
        args.max_tokens,
        compress_loops=args.compress_loops,
    )

    # Generate output suffix and paths
    judge_tag = "geministd" if args.judge == "gemini-standard" else None
    suffix = generate_output_suffix(args.run_id, judge_tag=judge_tag)
    output_dir = results_path.parent
    output_dir.mkdir(exist_ok=True)

    # Save results and config
    out_results_path = save_results(output_dir, suffix, analyzed_results)
    config_path = save_config(output_dir, suffix, args, stats)

    # Print summary
    print()
    print("=" * 60)
    print("BACKTRACKING ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total responses: {stats['total_responses']}")
    print(f"Successfully parsed: {stats['parsed_count']}")
    print(f"Parse failures: {stats['parse_failures']}")
    print()
    print("Backtracking classification distribution:")
    total = stats["parsed_count"]
    for label in ("normal", "backtracking", "nonsensical"):
        count = stats["distribution"][label]
        pct = 100 * count / total if total > 0 else 0
        bar = "\u2588" * int(pct / 2)
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    print()
    print("Correctness distribution:")
    corr = stats["correctness"]
    corr_total = corr["correct"] + corr["incorrect"] + corr["no_answer"]
    for label in ("correct", "incorrect", "no_answer"):
        count = corr[label]
        pct = 100 * count / corr_total if corr_total > 0 else 0
        bar = "\u2588" * int(pct / 2)
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    print()
    print(f"Results saved to: {out_results_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
