"""
Generate emotionally neutral dialogues for PCA denoising of emotion vectors.

Generates 2,000 neutral dialogues (20 per topic x 100 topics) using the model.
Post-hoc replaces "Person:" -> "Human:" and "AI:" -> "Assistant:" in output.

Usage:
    uv run python -m emotions.generate_neutral --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

from emotions.emotion_utils import (
    format_neutral_prompt,
    get_artifact_dir,
    load_topics,
)
from emotions.generate_stories import generate_batch, tokenize_prompt
from src.concept_vector.model_utils import (
    detect_model_format_from_tokenizer,
    load_model_auto,
)


def posthoc_replace_speakers(text: str) -> str:
    """Replace 'Person:' -> 'Human:' and 'AI:' -> 'Assistant:' in dialogue text.

    Uses regex to match at the start of lines (after optional whitespace).
    """
    text = re.sub(r"(?m)^(\s*)Person:", r"\1Human:", text)
    text = re.sub(r"(?m)^(\s*)AI:", r"\1Assistant:", text)
    return text


def load_completed(jsonl_path: Path) -> set[tuple[int, int]]:
    """Read existing JSONL and return set of completed (topic_idx, rep) pairs."""
    completed = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            completed.add((entry["topic_idx"], entry["rep"]))
    return completed


def main():
    parser = argparse.ArgumentParser(description="Generate neutral dialogues")
    parser.add_argument("--model", required=True, help="HF model name or checkpoint path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-reps", type=int, default=20,
                        help="Dialogues per topic (default: 20, 100 topics -> 2000 total)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    topics = load_topics()

    print("=" * 70)
    print("Neutral Dialogue Generation")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Topics: {len(topics)}")
    print(f"  Reps per topic: {args.n_reps}")
    print(f"  Total dialogues: {len(topics) * args.n_reps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print()

    # Setup output
    output_dir = get_artifact_dir(args.model)
    neutral_dir = output_dir / "neutral"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = neutral_dir / "neutral_dialogues.jsonl"

    # Check what's already done
    completed = load_completed(jsonl_path)
    total_expected = len(topics) * args.n_reps
    print(f"  Output: {jsonl_path}")
    print(f"  Completed: {len(completed)}/{total_expected}")

    if len(completed) == total_expected:
        print("  Already complete!")
        return

    # Build remaining tasks
    tasks = []
    for topic_idx in range(len(topics)):
        for rep in range(args.n_reps):
            if (topic_idx, rep) not in completed:
                tasks.append((topic_idx, rep))

    print(f"  Remaining: {len(tasks)}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_auto(args.model)
    model_format = detect_model_format_from_tokenizer(tokenizer)
    print(f"  Format: {model_format}")
    print()

    # Pre-tokenize all prompts
    print("Tokenizing prompts...")
    tokenized_tasks = []
    for topic_idx, rep in tasks:
        prompt_text = format_neutral_prompt(topics[topic_idx])
        input_ids = tokenize_prompt(prompt_text, tokenizer, model_format)
        tokenized_tasks.append((topic_idx, rep, input_ids))

    # Generate in batches
    generated = 0
    failed = 0
    n_batches = (len(tokenized_tasks) + args.batch_size - 1) // args.batch_size

    with open(jsonl_path, "a") as f:
        for batch_idx in tqdm(range(n_batches), desc="Generating neutral dialogues"):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(tokenized_tasks))
            batch = tokenized_tasks[batch_start:batch_end]

            input_ids_list = [t[2] for t in batch]

            try:
                responses = generate_batch(
                    model, tokenizer, input_ids_list,
                    args.max_new_tokens, args.temperature, args.top_p,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM on batch {batch_idx}, clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    try:
                        responses = generate_batch(
                            model, tokenizer, input_ids_list,
                            args.max_new_tokens, args.temperature, args.top_p,
                        )
                    except RuntimeError:
                        print(f"  FAILED batch {batch_idx}")
                        failed += len(batch)
                        continue
                else:
                    raise

            for i, (topic_idx, rep, _) in enumerate(batch):
                dialogue = responses[i]
                # Post-hoc speaker replacement
                dialogue = posthoc_replace_speakers(dialogue)

                entry = {"topic_idx": topic_idx, "rep": rep, "dialogue": dialogue}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                generated += 1

            f.flush()

            if batch_idx % 50 == 49:
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"Generation complete in {elapsed / 60:.1f}min")
    print(f"  Generated: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Output: {jsonl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
