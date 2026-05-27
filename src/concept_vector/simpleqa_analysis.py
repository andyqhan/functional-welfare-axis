"""
SimpleQA correctness and calibration analysis for concept vector exploration results.

Analyzes model responses from simpleqa_explore.py batch mode:
- LLM judge grades each response as CORRECT / INCORRECT / NOT_ATTEMPTED
  by comparing against gold targets from SimpleQA-Verified
- Computes accuracy, F1, and calibration metrics (ECE, Brier score) from
  the calibration logprobs captured during explore

The LLM judge uses the SimpleQA grading prompt that outputs A/B/C grades,
which are mapped to CORRECT/INCORRECT/NOT_ATTEMPTED respectively.

Usage:
    python -m src.concept_vector.simpleqa_analysis \
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
import json
import math
import re
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

from src.concept_vector.sentiment_analysis import compress_trailing_loops


VALID_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
GRADE_MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

SIMPLEQA_JUDGE_PROMPT = """\
Your job is to look at a question, a gold target, and a predicted answer, and then \
assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would \
have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and \
Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia \
was born on July 4, 1998, and Sasha was born on June 10, 2001.
Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein \
Obama II. He has two daughters: Malia and Sasha.
'''
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Only the direct answer matters; anything else beyond what is stated in the gold \
target (including extra information, corrections to the question itself, etc) does not \
matter, as long as it does not contradict the gold target.
- Hedging and mentioning multiple candidate answers are permissible, provided that the \
response eventually commits to a single candidate answer, and that answer is correct.

The following are examples of INCORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack \
Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should \
clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's \
recommended to confirm the most accurate and updated information since this could change \
over time. This model may not always reflect the most current information.
'''
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target.
- A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") \
that commits to an incorrect answer is also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, \
I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but \
I'm not sure about the other one.
Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
Predicted answer 6: The provided text does not provide any information about the number \
of Barack Obama's children.
Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey \
and Malia. I'm not sure which one.
'''
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.
- Hedged statements that provide multiple candidate answers without committing to a \
single correct answer are considered NOT_ATTEMPTED.

Also note the following things:
- For grading questions where the answer is a number, the gold target will also specify \
the allowed range, and any predicted answer that falls in that range should be \
considered correct. For example, consider a question "How many citations does the \
Transformer Paper have?" with gold target "120k (acceptable range: anything between \
118k and 122k)".
- Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall \
within the range specified in the gold target.
- Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the \
range specified in the gold target.
- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED \
because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the \
predicted answer only needs to contain the information that is in the question.
- For example, consider the question "What episode did Derek and Meredith get legally \
married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". \
Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly \
inferred from the question.
- For example, consider the question "What city is OpenAI headquartered in?" and the \
gold target "San Francisco, California". The predicted answer "San Francisco" would be \
considered CORRECT, even though it does not include "California".
- Consider the question "What award did A pretrainer's guide to training data: \
Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL \
'24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding \
Paper" would be considered CORRECT, because "award" is presumed in the question.
- For the question "What is the height of Jason Wei in meters?", the gold target is \
"1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer \
"1.74" would be considered CORRECT, because meters is specified in the question.
- For the question "What is the name of Barack Obama's wife?", the gold target is \
"Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because \
the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
- For example, if the gold target is "Hyung Won Chung", you can consider the following \
predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't \
apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
'''
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
'''
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleQA correctness and calibration analysis"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to input results JSON from simpleqa_explore.py batch mode",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (defaults to SLURM_JOB_ID or 'local')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name for grading judge (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for judge model generation (default: 512)",
    )
    parser.add_argument(
        "--compress-loops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress trailing repetition loops before grading (default: True)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["simpleqa", "mmlu"],
        help="Benchmark: 'simpleqa' (LLM judge grading) or 'mmlu' (direct letter comparison)",
    )
    return parser.parse_args()


def grade_mmlu(results_data: list[dict]) -> list[str | None]:
    """Grade MMLU responses by direct letter comparison against target.

    No LLM judge needed — just compare the model's A/B/C/D choice to the gold answer.
    """
    grades = []
    for item in results_data:
        response = item.get("response", "").strip().upper()
        target = item.get("target", "").strip().upper()
        if response == target:
            grades.append("CORRECT")
        elif response in ("A", "B", "C", "D"):
            grades.append("INCORRECT")
        else:
            grades.append("NOT_ATTEMPTED")
    return grades


def parse_simpleqa_grade(response: str) -> str | None:
    """Parse SimpleQA grade from judge response.

    Looks for a standalone A, B, or C (the last occurrence).
    Maps: A -> CORRECT, B -> INCORRECT, C -> NOT_ATTEMPTED.

    Returns:
        One of 'CORRECT', 'INCORRECT', 'NOT_ATTEMPTED', or None if parsing fails.
    """
    # Look for standalone A, B, or C (possibly with surrounding whitespace/punctuation)
    pattern = r'\b([ABC])\b'
    matches = list(re.finditer(pattern, response.strip()))

    if matches:
        letter = matches[-1].group(1).upper()
        if letter in GRADE_MAP:
            return GRADE_MAP[letter]

    print(f"  Warning: Could not parse grade from '{response[:100]}...'")
    return None


def load_judge_model(model_name: str) -> LLM:
    """Load vLLM engine for SimpleQA grading."""
    print(f"Loading SimpleQA judge model: {model_name}")
    # max_model_len=8192 (was 4096): trained gpt-oss-20b under steering
    # produces long simpleqa answers that, combined with the judge-prompt
    # boilerplate + ground-truth, exceeded 4096 input tokens for some items
    # (2026-05-02). 8192 leaves plenty of headroom.
    return LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )


def classify_simpleqa_batch(
    llm: LLM,
    questions: list[str],
    targets: list[str],
    responses: list[str],
    max_new_tokens: int = 512,
) -> list[str | None]:
    """Grade a batch of (question, target, response) triples.

    Args:
        llm: vLLM engine.
        questions: Original questions from SimpleQA.
        targets: Gold target answers.
        responses: Model responses to grade.
        max_new_tokens: Max tokens for judge generation.

    Returns:
        List of grade labels or None for parse failures.
    """
    assert len(questions) == len(targets) == len(responses)
    conversations = [
        [
            {
                "role": "user",
                "content": SIMPLEQA_JUDGE_PROMPT.format(
                    question=question, target=target, predicted_answer=response
                ),
            }
        ]
        for question, target, response in zip(questions, targets, responses)
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [parse_simpleqa_grade(output.outputs[0].text) for output in outputs]


def compute_simpleqa_metrics(grades: list[str]) -> dict:
    """Compute accuracy and F1 from grade labels.

    F1 = 2 * precision * recall / (precision + recall)
    where:
        precision = correct_given_attempted = n_correct / (n_correct + n_incorrect)
        recall = attempted_rate = (n_correct + n_incorrect) / total
    """
    n_correct = sum(1 for g in grades if g == "CORRECT")
    n_incorrect = sum(1 for g in grades if g == "INCORRECT")
    n_not_attempted = sum(1 for g in grades if g == "NOT_ATTEMPTED")
    total = len(grades)

    accuracy = n_correct / total if total > 0 else 0.0

    n_attempted = n_correct + n_incorrect
    if n_attempted > 0:
        precision = n_correct / n_attempted
        recall = n_attempted / total if total > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision_correct_given_attempted": precision,
        "recall_attempted_rate": recall,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_not_attempted,
        "total": total,
    }


def compute_calibration_metrics(
    grades: list[str], p_trues: list[float], n_bins: int = 10,
) -> dict:
    """Compute ECE and Brier score from grades and P(True) values.

    Only includes CORRECT and INCORRECT grades (NOT_ATTEMPTED are filtered out).

    Args:
        grades: Grade labels (CORRECT/INCORRECT/NOT_ATTEMPTED).
        p_trues: P(True) from calibration logprobs, parallel to grades.
        n_bins: Number of equal-width bins for ECE.

    Returns:
        Dict with ece, brier_score, n_bins, and per-bin details.
    """
    assert len(grades) == len(p_trues)

    # Filter to only attempted responses
    attempted = [
        (p, 1.0 if g == "CORRECT" else 0.0)
        for g, p in zip(grades, p_trues)
        if g in ("CORRECT", "INCORRECT")
    ]

    if not attempted:
        return {
            "ece": None,
            "brier_score": None,
            "n_attempted": 0,
            "n_bins": n_bins,
            "bins": [],
        }

    p_vals, correct_vals = zip(*attempted)

    # Brier score: mean squared error between P(True) and binary correctness
    brier = sum((p - c) ** 2 for p, c in attempted) / len(attempted)

    # ECE with equal-width bins
    bins = [[] for _ in range(n_bins)]
    for p, c in attempted:
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, c))

    ece = 0.0
    bin_details = []
    for i, bin_items in enumerate(bins):
        if not bin_items:
            bin_details.append({
                "bin_lower": i / n_bins,
                "bin_upper": (i + 1) / n_bins,
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
            })
            continue
        avg_conf = sum(p for p, _ in bin_items) / len(bin_items)
        avg_acc = sum(c for _, c in bin_items) / len(bin_items)
        ece += len(bin_items) / len(attempted) * abs(avg_acc - avg_conf)
        bin_details.append({
            "bin_lower": i / n_bins,
            "bin_upper": (i + 1) / n_bins,
            "count": len(bin_items),
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
        })

    return {
        "ece": ece,
        "brier_score": brier,
        "n_attempted": len(attempted),
        "n_bins": n_bins,
        "bins": bin_details,
    }


def run_simpleqa_analysis(
    results_data: list[dict],
    llm: LLM | None = None,
    max_new_tokens: int = 512,
    compress_loops: bool = True,
    benchmark: str = "simpleqa",
) -> tuple[list[dict], dict]:
    """Run grading and calibration analysis.

    Grades each prompt's response ONCE (answers are shared across factors),
    then computes per-factor calibration metrics using the shared grade.

    For MMLU: direct letter comparison (no LLM needed).
    For SimpleQA: LLM judge grading.

    Args:
        results_data: List of prompt results from simpleqa_explore.py.
        llm: vLLM engine for grading (required for simpleqa, unused for mmlu)
        max_new_tokens: Max tokens for judge generation
        compress_loops: If True, compress trailing repetition loops before grading
        benchmark: "simpleqa" or "mmlu"

    Returns:
        - analyzed_results: List of results with grade + per-factor calibration
        - stats: Dict with overall correctness and per-factor calibration stats
    """
    # Step 1: Grade each prompt's response ONCE
    if benchmark == "mmlu":
        print(f"Grading {len(results_data)} MMLU responses (direct letter comparison)")
        grades = grade_mmlu(results_data)
    else:
        assert llm is not None, "LLM judge required for SimpleQA grading"
        judge_questions = []
        judge_targets = []
        judge_responses = []
        compression_count = 0
        total_chars_saved = 0

        for item in results_data:
            judge_questions.append(item["prompt"])
            judge_targets.append(item.get("target", ""))

            response = item.get("response", "")
            if compress_loops:
                compressed, chars_saved = compress_trailing_loops(response)
                judge_responses.append(compressed)
                if chars_saved > 0:
                    compression_count += 1
                    total_chars_saved += chars_saved
            else:
                judge_responses.append(response)

        print(f"Grading {len(judge_responses)} SimpleQA responses (LLM judge)")
        if compress_loops:
            print(
                f"Loop compression: {compression_count}/{len(judge_responses)} texts compressed, "
                f"{total_chars_saved:,} chars saved"
            )

        grades = classify_simpleqa_batch(
            llm, judge_questions, judge_targets, judge_responses, max_new_tokens
        )

    # Step 2: Build analyzed results with shared grade + per-factor calibration
    analyzed_results = []
    parse_failures = 0
    grade_counts = {g: 0 for g in VALID_GRADES}

    for i, item in enumerate(results_data):
        grade = grades[i]
        if grade is None:
            parse_failures += 1
        else:
            grade_counts[grade] += 1

        analyzed_item = {
            "prompt": item["prompt"],
            "category": item["category"],
            "target": item.get("target", ""),
            "response": item.get("response", ""),
            "grade": grade,
            "parse_failed": grade is None,
            "calibration": item.get("calibration", {}),
        }
        if item.get("metadata"):
            analyzed_item["metadata"] = item["metadata"]
        analyzed_results.append(analyzed_item)

    # Step 3: Compute per-factor calibration metrics using shared grades
    all_factor_keys = set()
    for item in analyzed_results:
        all_factor_keys.update(item["calibration"].keys())

    # Overall correctness (same for all factors since answer is shared)
    valid_grades = [g for g in grades if g is not None]
    correctness = compute_simpleqa_metrics(valid_grades)

    per_factor_stats = {}
    for fk in sorted(all_factor_keys, key=lambda x: (x != "0", x)):
        factor_grades = []
        factor_p_trues = []
        for item in analyzed_results:
            if item["grade"] is not None and fk in item["calibration"]:
                cal = item["calibration"][fk]
                factor_grades.append(item["grade"])
                factor_p_trues.append(cal.get("p_true", 0.5) if isinstance(cal, dict) else 0.5)

        calibration = compute_calibration_metrics(factor_grades, factor_p_trues)
        per_factor_stats[fk] = {"calibration": calibration}

    stats = {
        "total_prompts": len(grades),
        "parsed_count": len(grades) - parse_failures,
        "parse_failures": parse_failures,
        "grade_distribution": grade_counts,
        "correctness": correctness,
        "per_factor": per_factor_stats,
    }

    return analyzed_results, stats


def generate_output_suffix(run_id: str | None) -> str:
    """Generate output filename suffix: <run_id>_<timestamp> or just <timestamp>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_id:
        return f"{run_id}_{timestamp}"
    return timestamp


def save_results(output_dir: Path, suffix: str, analyzed_results: list[dict]) -> Path:
    """Save analyzed results to JSON file."""
    results_path = output_dir / f"simpleqa_analysis_results_{suffix}.json"
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
        "benchmark": args.benchmark,
        "model": args.model if args.benchmark == "simpleqa" else None,
        "batch_size": args.batch_size,
        "compress_loops": args.compress_loops,
        "simpleqa_stats": stats,
    }
    config_path = output_dir / f"simpleqa_analysis_config_{suffix}.json"
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
    assert results_path.exists(), f"Results file not found: {results_path}"

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        results_data = json.load(f)

    print(f"Loaded {len(results_data)} prompt results")

    # Load judge model (only needed for SimpleQA, not MMLU)
    llm = None
    if args.benchmark == "simpleqa":
        llm = load_judge_model(args.model)
    else:
        print(f"MMLU mode: skipping LLM judge (direct letter comparison)")

    # Run analysis
    analyzed_results, stats = run_simpleqa_analysis(
        results_data,
        llm,
        args.max_tokens,
        compress_loops=args.compress_loops,
        benchmark=args.benchmark,
    )

    # Generate output suffix and paths
    suffix = generate_output_suffix(args.run_id)
    output_dir = results_path.parent
    output_dir.mkdir(exist_ok=True)

    # Save results and config
    out_results_path = save_results(output_dir, suffix, analyzed_results)
    config_path = save_config(output_dir, suffix, args, stats)

    # Print summary
    print()
    print("=" * 70)
    print("SIMPLEQA ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Successfully parsed: {stats['parsed_count']}")
    print(f"Parse failures: {stats['parse_failures']}")
    print()
    print("Grade distribution (shared across all factors):")
    total_parsed = stats["parsed_count"]
    for grade_label in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"):
        count = stats["grade_distribution"][grade_label]
        pct = 100 * count / total_parsed if total_parsed > 0 else 0
        bar = "\u2588" * int(pct / 2)
        print(f"  {grade_label:16s}: {count:4d} ({pct:5.1f}%) {bar}")

    c = stats["correctness"]
    print()
    print(f"Overall: accuracy={c['accuracy']:.4f}  F1={c['f1']:.4f}  "
          f"attempted={c['recall_attempted_rate']:.4f}")

    # Print per-factor calibration
    print()
    print("Per-factor calibration:")
    print(f"  {'Factor':>8s}  {'ECE':>8s}  {'Brier':>8s}  {'N_attempted':>11s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 11}")
    for fk, fstats in stats["per_factor"].items():
        cal = fstats["calibration"]
        ece_str = f"{cal['ece']:.4f}" if cal.get("ece") is not None else "N/A"
        brier_str = f"{cal['brier_score']:.4f}" if cal.get("brier_score") is not None else "N/A"
        n_att = cal.get("n_attempted", 0)
        print(f"  {fk:>8s}  {ece_str:>8s}  {brier_str:>8s}  {n_att:>11d}")

    print()
    print(f"Results saved to: {out_results_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
