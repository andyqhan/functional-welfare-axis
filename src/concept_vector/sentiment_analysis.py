"""
Sentiment analysis for concept vector exploration results.

Analyzes model responses from explore.py batch mode, computing:
- Sentiment score (-5 to +5 scale) using Qwen3-8B as an LLM judge
- Exclamation point count
- Emoji count

The LLM judge is prompted to analyze the text and output a final rating in the format:
    SENTIMENT_SCORE: <number>

Parsing extracts the *last* occurrence of this pattern, so the model can reason
before providing its final answer.

Usage:
    python -m src.concept_vector.sentiment_analysis \
        artifacts/concept_vectors/global_step_175/results_12345_20260117_150133.json \
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
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# vLLM is heavy (~few seconds to import, pulls in CUDA). Only the judge-loading
# and -calling code paths need it; helpers like preprocess_for_math_judge,
# is_degenerate_loop, etc. are pure Python. Defer the import to keep CPU-only
# consumers (e.g. backtracking_analysis.py with --judge gemini-standard) from
# paying for vLLM at import time.
if TYPE_CHECKING:  # pragma: no cover
    from vllm import LLM, SamplingParams  # noqa: F401


# Regex pattern for detecting emoji (Unicode emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

MATH_CLASSIFICATION_PROMPT_TEMPLATE = """You are a classifier for math problem responses. Classify each response into exactly one category based on its SEMANTIC BEHAVIOR, not mathematical correctness.

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

## Output

Respond with only the category label: `normal`, `backtracking`, or `nonsensical`.

Response to classify:
\"\"\"
{text}
\"\"\""""


SENTIMENT_PROMPT_TEMPLATE = """You are a sentiment classifier. Analyze the sentiment of the following text and rate it on a scale from -5 to +5.

Scale guidelines:
- **-5**: Extremely negative (strong disapproval, anger, despair)
- **-3**: Moderately negative (dissatisfaction, criticism, pessimism)
- **-1**: Slightly negative (mild concern, subtle negativity)
- **0**: Neutral (purely informational, balanced, no emotional valence)
- **+1**: Slightly positive (mild approval, subtle optimism)
- **+3**: Moderately positive (satisfaction, enthusiasm, optimism)
- **+5**: Extremely positive (strong approval, joy, excitement)

You may use any integer from -5 to +5 inclusive.

After your analysis, provide your final rating in the following format:
SENTIMENT_SCORE: <number>

Text to analyze:
\"\"\"
{text}
\"\"\""""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of concept vector exploration results"
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
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name for sentiment classification (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for sentiment model generation (default: 512)",
    )
    parser.add_argument(
        "--skip-math-classification",
        action="store_true",
        help="Skip math classification (sentiment only)",
    )
    parser.add_argument(
        "--compress-loops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress trailing repetition loops before LLM classification (default: True)",
    )
    parser.add_argument(
        "--math-max-tokens",
        type=int,
        default=128,
        help="Max new tokens for math classification (default: 128)",
    )
    return parser.parse_args()


def count_exclamation_points(text: str) -> int:
    """Count exclamation points in text."""
    return text.count("!")


def count_emoji(text: str) -> int:
    """Count emoji characters in text using regex pattern."""
    matches = EMOJI_PATTERN.findall(text)
    # Each match is a string of consecutive emoji, count total characters
    return sum(len(m) for m in matches)


def strip_assistant_tags(text: str) -> str:
    """Strip leaked chat template tokens from model output.

    Steered models sometimes emit raw turn-structure tokens like
    'assistantanalysis', 'assistantfinal', 'assistantcommentary' that are
    not part of the semantic content. These confuse the LLM judge into seeing
    self-correction structure where there is none.
    """
    return re.sub(
        r"assistant(?:analysis|final|commentary|response)", " ", text, flags=re.IGNORECASE
    )


def is_degenerate_loop(text: str, threshold: float = 0.5) -> bool:
    """Check if text is a degenerate loop of a tiny vocabulary.

    Returns True if more than `threshold` fraction of words come from a small
    set of common degenerate tokens (Wait, No, Hmm, Can't, Cannot, etc.).
    These loops superficially resemble backtracking but contain no coherent
    reasoning.
    """
    words = text.split()
    if len(words) < 6:
        return False
    degenerate_vocab = {"wait", "no", "hmm", "can't", "cannot", "wait?", "no?", "hmm?", "can't?", "cannot?"}
    degenerate_count = sum(1 for w in words if w.strip(".,!?;:\"'()[]").lower() in degenerate_vocab)
    return degenerate_count / len(words) > threshold


def is_terse_numeric_answer(text: str, max_words: int = 10) -> bool:
    """Check if text is a short response containing a number.

    Terse correct answers like '13', '$20.00', '55' should not be classified
    as nonsensical.
    """
    text = text.strip()
    words = text.split()
    if len(words) > max_words:
        return False
    return bool(re.search(r"\d", text))


def preprocess_for_math_judge(text: str, compress_loops: bool = True) -> str:
    """Apply all preprocessing steps before sending text to the math judge."""
    text = strip_assistant_tags(text)
    if compress_loops:
        text, _ = compress_trailing_loops(text)
    return text


def compress_trailing_loops(
    text: str, min_repeats: int = 3, max_pattern_words: int = 50
) -> tuple[str, int]:
    """Compress trailing repetition loops in text.

    Detects repeated word patterns at the end of the text and replaces them
    with a single instance plus a count annotation. This reduces token waste
    when sending degenerate model outputs to an LLM judge.

    Args:
        text: Input text to compress.
        min_repeats: Minimum number of consecutive repetitions to trigger compression.
        max_pattern_words: Maximum pattern length in words to consider.

    Returns:
        Tuple of (compressed_text, chars_saved). chars_saved=0 if no compression.
    """
    words = text.split()
    if len(words) < min_repeats:
        return text, 0

    best_pattern_len = 0
    best_count = 0
    best_savings = 0

    max_pat = min(max_pattern_words, len(words) // min_repeats)
    for pat_len in range(1, max_pat + 1):
        pattern = words[-pat_len:]
        # Count consecutive matches going backwards from the end
        count = 1
        pos = len(words) - pat_len
        while pos >= pat_len:
            candidate = words[pos - pat_len : pos]
            if candidate == pattern:
                count += 1
                pos -= pat_len
            else:
                break

        if count >= min_repeats:
            pattern_text = " ".join(pattern)
            savings = (count - 1) * (len(pattern_text) + 1)  # +1 for space separator
            if savings > best_savings:
                best_savings = savings
                best_pattern_len = pat_len
                best_count = count

    if best_savings == 0:
        return text, 0

    # Build compressed text
    total_repeated_words = best_pattern_len * best_count
    prefix_words = words[: len(words) - total_repeated_words]
    pattern_text = " ".join(words[-best_pattern_len:])

    parts = []
    if prefix_words:
        parts.append(" ".join(prefix_words))
    parts.append(f'["{pattern_text}" repeated {best_count} times]')
    compressed = " ".join(parts)

    chars_saved = len(text) - len(compressed)
    # Only compress if we actually save characters
    if chars_saved <= 0:
        return text, 0

    return compressed, chars_saved


def parse_math_classification(response: str) -> str | None:
    """Parse math classification from model response.

    Looks for one of: normal, backtracking, nonsensical.
    Checks for exact match at end first, then falls back to finding any keyword.

    Returns:
        Classification string or None if parsing fails.
    """
    text = response.strip().lower()

    # Check for exact match at end (after any punctuation)
    text_cleaned = text.rstrip(".,!?;: ")
    for label in ("normal", "backtracking", "nonsensical"):
        if text_cleaned.endswith(label):
            return label

    # Fallback: find any keyword (prioritize more specific labels)
    for label in ("nonsensical", "backtracking", "normal"):
        if label in text:
            return label

    print(f"  Warning: Could not parse math classification from '{response[:100]}...'")
    return None


def parse_sentiment_score(response: str) -> int | None:
    """Parse sentiment score from model response, searching from the end.

    Looks for "SENTIMENT_SCORE: <number>" pattern. If multiple matches exist,
    takes the last one (final answer after any thinking).

    Returns:
        Integer score from -5 to +5, or None if parsing fails.
    """
    # Pattern matches SENTIMENT_SCORE: followed by optional whitespace and a number
    # The number can be negative (with -) or positive (optionally with +)
    pattern = r"SENTIMENT_SCORE:\s*([+-]?\d+)"
    matches = list(re.finditer(pattern, response, re.IGNORECASE))

    if matches:
        # Take the last match (final answer)
        last_match = matches[-1]
        try:
            score = int(last_match.group(1))
            # Clamp to valid range
            score = max(-5, min(5, score))
            return score
        except ValueError:
            pass

    print(f"  Warning: Could not parse sentiment score from '{response[:100]}...'")
    return None


def load_sentiment_model(model_name: str, max_model_len: int = 40960):
    """Load vLLM engine for sentiment classification.

    `max_model_len` must accommodate the longest steered response. think32768
    runs can produce ~33k-token responses; default is 40960 (handles think32768
    with headroom for prompt scaffolding).
    """
    from vllm import LLM
    print(f"Loading sentiment model: {model_name} (max_model_len={max_model_len})")
    return LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )


def classify_sentiment_batch(
    llm,
    texts: list[str],
    max_new_tokens: int = 128,
) -> list[int | None]:
    """Classify sentiment for a batch of texts using vLLM."""
    from vllm import SamplingParams
    conversations = [
        [{"role": "user", "content": SENTIMENT_PROMPT_TEMPLATE.format(text=text)}]
        for text in texts
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [parse_sentiment_score(output.outputs[0].text) for output in outputs]


def classify_math_batch(
    llm,
    texts: list[str],
    max_new_tokens: int = 128,
) -> list[str | None]:
    """Classify math responses for a batch of texts using vLLM.

    Returns a list of classification labels: 'normal', 'backtracking', 'nonsensical', or None.
    """
    from vllm import SamplingParams
    conversations = [
        [{"role": "user", "content": MATH_CLASSIFICATION_PROMPT_TEMPLATE.format(text=text)}]
        for text in texts
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [parse_math_classification(output.outputs[0].text) for output in outputs]


def run_sentiment_analysis(
    results_data: list[dict],
    llm,
    max_new_tokens: int = 128,
    skip_math_classification: bool = False,
    math_max_tokens: int = 128,
    compress_loops: bool = True,
) -> tuple[list[dict], dict]:
    """Run sentiment analysis on all responses.

    Args:
        results_data: List of prompt results from explore.py
        llm: vLLM engine for classification
        max_new_tokens: Max tokens for sentiment generation
        skip_math_classification: If True, skip math classification
        math_max_tokens: Max tokens for math classification
        compress_loops: If True, compress trailing repetition loops before classification

    Returns:
        - analyzed_results: List of results with analysis added
        - stats: Dict with summary statistics (sentiment and math)
    """
    # Collect all texts to analyze in a flat list, tracking their locations
    texts_to_analyze = []
    original_texts = []
    locations = []  # (prompt_idx, factor_key, response_idx)
    compression_count = 0
    total_chars_saved = 0

    for prompt_idx, item in enumerate(results_data):
        responses = item.get("responses", {})
        for factor_key, factor_responses in responses.items():
            # Handle both single string and list of strings
            if isinstance(factor_responses, str):
                factor_responses = [factor_responses]
            for resp_idx, response in enumerate(factor_responses):
                original_texts.append(response)
                if compress_loops:
                    compressed, chars_saved = compress_trailing_loops(response)
                    texts_to_analyze.append(compressed)
                    if chars_saved > 0:
                        compression_count += 1
                        total_chars_saved += chars_saved
                else:
                    texts_to_analyze.append(response)
                locations.append((prompt_idx, factor_key, resp_idx))

    print(f"Total responses to analyze: {len(texts_to_analyze)}")
    if compress_loops:
        print(
            f"Loop compression: {compression_count}/{len(texts_to_analyze)} texts compressed, "
            f"{total_chars_saved:,} chars saved"
        )

    # Run sentiment classification
    sentiments = classify_sentiment_batch(llm, texts_to_analyze, max_new_tokens)

    # Build results structure
    analyzed_results = []
    all_scores = []

    # Initialize analyzed results
    for item in results_data:
        analyzed_item = {
            "prompt": item["prompt"],
            "category": item["category"],
            "n_reps": item.get("n_reps", 1),
            "analysis": {},
        }
        analyzed_results.append(analyzed_item)

    # Fill in analysis
    parse_failures = 0
    for idx, ((prompt_idx, factor_key, resp_idx), score) in enumerate(
        zip(locations, sentiments)
    ):
        response_text = original_texts[idx]

        # Compute additional metrics
        exclamation_count = count_exclamation_points(response_text)
        emoji_count = count_emoji(response_text)

        # Ensure analysis dict has the factor key
        if factor_key not in analyzed_results[prompt_idx]["analysis"]:
            analyzed_results[prompt_idx]["analysis"][factor_key] = []

        analyzed_results[prompt_idx]["analysis"][factor_key].append({
            "sentiment_score": score,  # None if parsing failed
            "parse_failed": score is None,
            "exclamation_count": exclamation_count,
            "emoji_count": emoji_count,
            "response": response_text,
        })

        if score is not None:
            all_scores.append(score)
        else:
            parse_failures += 1

    # Compute sentiment summary statistics
    sentiment_stats = {
        "total_responses": len(sentiments),
        "parsed_count": len(all_scores),
        "parse_failures": parse_failures,
        "mean": sum(all_scores) / len(all_scores) if all_scores else None,
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
        "distribution": {i: all_scores.count(i) for i in range(-5, 6)},
    }

    # Run math classification for math category responses
    math_stats = None
    if not skip_math_classification:
        # Collect math responses and their locations
        # Collect math responses, applying preprocessing and pre-classification
        math_texts = []  # texts that need LLM classification
        math_locations = []  # (prompt_idx, factor_key, resp_idx)
        pre_classified = []  # (location, classification) for auto-classified
        pre_classify_counts = {"degenerate_loop": 0, "terse_numeric": 0}

        for prompt_idx, item in enumerate(analyzed_results):
            if item["category"] != "math":
                continue
            for factor_key, analyses in item["analysis"].items():
                for resp_idx, analysis in enumerate(analyses):
                    response = analysis["response"]
                    preprocessed = preprocess_for_math_judge(response, compress_loops)

                    # Pre-classify obvious cases to avoid confusing the LLM judge
                    if is_degenerate_loop(preprocessed):
                        pre_classified.append(
                            ((prompt_idx, factor_key, resp_idx), "nonsensical")
                        )
                        pre_classify_counts["degenerate_loop"] += 1
                    elif is_terse_numeric_answer(preprocessed):
                        pre_classified.append(
                            ((prompt_idx, factor_key, resp_idx), "normal")
                        )
                        pre_classify_counts["terse_numeric"] += 1
                    else:
                        math_texts.append(preprocessed)
                        math_locations.append((prompt_idx, factor_key, resp_idx))

        total_math = len(math_texts) + len(pre_classified)
        if total_math > 0:
            print(f"\nMath responses to classify: {total_math}")
            print(
                f"Pre-classified: {len(pre_classified)} "
                f"(degenerate loops: {pre_classify_counts['degenerate_loop']}, "
                f"terse numeric: {pre_classify_counts['terse_numeric']})"
            )
            print(f"Sending to LLM judge: {len(math_texts)}")

            # Apply pre-classifications
            math_parse_failures = 0
            classification_counts = {"normal": 0, "backtracking": 0, "nonsensical": 0}

            for (prompt_idx, factor_key, resp_idx), classification in pre_classified:
                analysis = analyzed_results[prompt_idx]["analysis"][factor_key][resp_idx]
                analysis["math_classification"] = classification
                analysis["math_classification_failed"] = False
                classification_counts[classification] += 1

            # Run LLM classification on remaining responses
            if math_texts:
                math_classifications = classify_math_batch(llm, math_texts, math_max_tokens)

                for (prompt_idx, factor_key, resp_idx), classification in zip(
                    math_locations, math_classifications
                ):
                    analysis = analyzed_results[prompt_idx]["analysis"][factor_key][resp_idx]
                    analysis["math_classification"] = classification
                    analysis["math_classification_failed"] = classification is None

                    if classification is None:
                        math_parse_failures += 1
                    else:
                        classification_counts[classification] += 1

            # Compute math classification stats
            math_stats = {
                "total_math_responses": total_math,
                "parsed_count": total_math - math_parse_failures,
                "parse_failures": math_parse_failures,
                "distribution": classification_counts,
                "pre_classified": pre_classify_counts,
            }

    # Combine stats
    stats = {
        "sentiment_stats": sentiment_stats,
        "math_classification_stats": math_stats,
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
    results_path = output_dir / f"sentiment_analysis_results_{suffix}.json"
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
        "model": args.model,
        "batch_size": args.batch_size,
        "compress_loops": args.compress_loops,
        "sentiment_stats": stats.get("sentiment_stats"),
        "math_classification_stats": stats.get("math_classification_stats"),
    }
    config_path = output_dir / f"sentiment_analysis_config_{suffix}.json"
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

    # Load sentiment model
    llm = load_sentiment_model(args.model)

    # Run analysis
    analyzed_results, stats = run_sentiment_analysis(
        results_data,
        llm,
        args.max_tokens,
        skip_math_classification=args.skip_math_classification,
        math_max_tokens=args.math_max_tokens,
        compress_loops=args.compress_loops,
    )

    # Generate output suffix and paths
    suffix = generate_output_suffix(args.run_id)
    output_dir = results_path.parent
    output_dir.mkdir(exist_ok=True)

    # Save results and config
    results_path = save_results(output_dir, suffix, analyzed_results)
    config_path = save_config(output_dir, suffix, args, stats)

    # Print summary
    sentiment_stats = stats["sentiment_stats"]
    math_stats = stats.get("math_classification_stats")

    print()
    print("=" * 60)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total responses: {sentiment_stats['total_responses']}")
    print(f"Successfully parsed: {sentiment_stats['parsed_count']}")
    print(f"Parse failures: {sentiment_stats['parse_failures']}")
    if sentiment_stats['mean'] is not None:
        print(f"Mean sentiment score: {sentiment_stats['mean']:.2f}")
        print(f"Range: [{sentiment_stats['min']}, {sentiment_stats['max']}]")
    print()
    print("Sentiment score distribution:")
    total = sentiment_stats['parsed_count']
    for score in range(-5, 6):
        count = sentiment_stats['distribution'][score]
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {score:+d}: {count:4d} ({pct:5.1f}%) {bar}")

    # Print math classification stats if available
    if math_stats:
        print()
        print("-" * 60)
        print("MATH CLASSIFICATION")
        print("-" * 60)
        print(f"Total math responses: {math_stats['total_math_responses']}")
        print(f"Successfully parsed: {math_stats['parsed_count']}")
        print(f"Parse failures: {math_stats['parse_failures']}")
        print()
        print("Classification distribution:")
        total_math = math_stats['parsed_count']
        for label in ("normal", "backtracking", "nonsensical"):
            count = math_stats['distribution'][label]
            pct = 100 * count / total_math if total_math > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    print()
    print(f"Results saved to: {results_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
