"""
Generate emotional stories for emotion concept vector extraction.

For each (emotion, topic, rep) triple, generates one short story using the model.
Stories are saved incrementally to JSONL files for crash-safe resume.

Usage:
    uv run python -m emotions.generate_stories --model Qwen/Qwen3-4B-Instruct-2507
    uv run python -m emotions.generate_stories --model Qwen/Qwen3-4B-Instruct-2507 \
        --emotions-start 0 --emotions-end 50 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from emotions.emotion_utils import (
    format_story_prompt,
    get_artifact_dir,
    load_emotions,
    load_topics,
)
from src.concept_vector.model_utils import (
    detect_model_format_from_tokenizer,
    format_generation_prompt_harmony,
    load_model_auto,
)


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


def tokenize_prompt(prompt_text: str, tokenizer, model_format: str) -> torch.Tensor:
    """Tokenize a generation prompt, returning input_ids of shape (seq_len,)."""
    if model_format == "harmony":
        messages = [{"role": "user", "content": prompt_text}]
        harmony_str = format_generation_prompt_harmony(messages)
        input_ids = tokenizer(
            harmony_str, return_tensors="pt", add_special_tokens=False,
        ).input_ids
    else:
        messages = [{"role": "user", "content": prompt_text}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
        )
    return input_ids[0]  # (seq_len,)


def generate_batch(
    model, tokenizer, input_ids_list: list[torch.Tensor],
    max_new_tokens: int, temperature: float, top_p: float,
) -> list[str]:
    """Generate responses for a batch of prompts. Returns list of decoded strings."""
    pad_token_id = tokenizer.pad_token_id

    # Left-pad to max length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded_ids = []
    attention_masks = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            padded = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.shape[0], dtype=torch.long)])
        else:
            padded = ids
            mask = torch.ones(ids.shape[0], dtype=torch.long)
        padded_ids.append(padded)
        attention_masks.append(mask)

    batch_input_ids = torch.stack(padded_ids).to(model.device)
    batch_attention_mask = torch.stack(attention_masks).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=pad_token_id,
        )

    # Decode each response (tokens after the input)
    responses = []
    for i in range(len(input_ids_list)):
        response_ids = output_ids[i][max_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


def generate_stories_for_emotion(
    model, tokenizer, emotion: str, topics: list[str],
    n_reps: int, output_dir: Path, model_format: str,
    batch_size: int, temperature: float, top_p: float,
    max_new_tokens: int, max_retries: int = 3,
) -> dict:
    """Generate all stories for one emotion. Returns stats dict."""
    stories_dir = output_dir / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = stories_dir / f"{emotion.replace(' ', '_')}.jsonl"

    # Check what's already done
    completed = load_completed(jsonl_path)
    total_expected = len(topics) * n_reps
    if len(completed) == total_expected:
        print(f"  [{emotion}] Already complete ({total_expected}/{total_expected})")
        return {"generated": 0, "skipped": total_expected, "failed": 0}

    # Build list of remaining tasks
    tasks = []
    for topic_idx in range(len(topics)):
        for rep in range(n_reps):
            if (topic_idx, rep) not in completed:
                tasks.append((topic_idx, rep))

    print(f"  [{emotion}] {len(completed)}/{total_expected} done, {len(tasks)} remaining")

    # Pre-tokenize all prompts
    tokenized_tasks = []
    for topic_idx, rep in tasks:
        prompt_text = format_story_prompt(emotion, topics[topic_idx])
        input_ids = tokenize_prompt(prompt_text, tokenizer, model_format)
        tokenized_tasks.append((topic_idx, rep, input_ids))

    # Generate in batches, append to JSONL
    generated = 0
    failed = 0
    n_batches = (len(tokenized_tasks) + batch_size - 1) // batch_size

    with open(jsonl_path, "a") as f:
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tokenized_tasks))
            batch = tokenized_tasks[batch_start:batch_end]

            input_ids_list = [t[2] for t in batch]

            # Generate with retries for the whole batch
            for attempt in range(max_retries):
                try:
                    responses = generate_batch(
                        model, tokenizer, input_ids_list,
                        max_new_tokens, temperature, top_p,
                    )
                    break
                except RuntimeError as e:
                    if attempt < max_retries - 1 and "out of memory" in str(e).lower():
                        print(f"    OOM on batch {batch_idx}, retrying with cache clear...")
                        torch.cuda.empty_cache()
                        continue
                    raise
            else:
                # All retries failed
                print(f"    FAILED batch {batch_idx} after {max_retries} retries")
                failed += len(batch)
                continue

            # Validate and write each response
            for i, (topic_idx, rep, _) in enumerate(batch):
                story = responses[i]
                # Validate: non-empty and at least a few words
                response_tokens = tokenizer.encode(story, add_special_tokens=False)
                if len(response_tokens) < 20:
                    # Retry this single prompt
                    for retry in range(max_retries):
                        single_response = generate_batch(
                            model, tokenizer, [input_ids_list[i]],
                            max_new_tokens, temperature, top_p,
                        )
                        story = single_response[0]
                        response_tokens = tokenizer.encode(story, add_special_tokens=False)
                        if len(response_tokens) >= 20:
                            break
                    else:
                        print(f"    Short story for topic={topic_idx} rep={rep}: {len(response_tokens)} tokens")
                        failed += 1
                        continue

                entry = {"topic_idx": topic_idx, "rep": rep, "story": story}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                generated += 1

            f.flush()

            # Periodic cache clear
            if batch_idx % 50 == 49:
                torch.cuda.empty_cache()

    return {"generated": generated, "skipped": len(completed), "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Generate emotional stories")
    parser.add_argument("--model", required=True, help="HF model name or checkpoint path")
    parser.add_argument("--emotions-start", type=int, default=0,
                        help="Start index into emotions list (for job splitting)")
    parser.add_argument("--emotions-end", type=int, default=None,
                        help="End index (exclusive) into emotions list")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-reps", type=int, default=12,
                        help="Stories per (emotion, topic) pair")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    emotions = load_emotions()
    topics = load_topics()
    emotions_slice = emotions[args.emotions_start : args.emotions_end]
    assert len(emotions_slice) > 0, (
        f"Empty emotions slice [{args.emotions_start}:{args.emotions_end}] "
        f"(total emotions: {len(emotions)})"
    )

    print("=" * 70)
    print("Emotion Story Generation")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Emotions: {len(emotions_slice)} ({args.emotions_start}:{args.emotions_end or len(emotions)})")
    print(f"  Topics: {len(topics)}")
    print(f"  Reps per (emotion, topic): {args.n_reps}")
    print(f"  Total stories per emotion: {len(topics) * args.n_reps}")
    print(f"  Total stories this job: {len(emotions_slice) * len(topics) * args.n_reps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_auto(args.model)
    model_format = detect_model_format_from_tokenizer(tokenizer)
    print(f"  Format: {model_format}")
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    print()

    # Setup output
    output_dir = get_artifact_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")
    print()

    # Generate for each emotion
    total_generated = 0
    total_skipped = 0
    total_failed = 0

    for emotion_idx, emotion in enumerate(emotions_slice):
        global_idx = args.emotions_start + emotion_idx
        print(f"\n[{emotion_idx + 1}/{len(emotions_slice)}] Emotion: '{emotion}' (global index {global_idx})")

        stats = generate_stories_for_emotion(
            model=model,
            tokenizer=tokenizer,
            emotion=emotion,
            topics=topics,
            n_reps=args.n_reps,
            output_dir=output_dir,
            model_format=model_format,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        total_generated += stats["generated"]
        total_skipped += stats["skipped"]
        total_failed += stats["failed"]

        elapsed = time.time() - t0
        emotions_done = emotion_idx + 1
        rate = elapsed / emotions_done
        remaining = (len(emotions_slice) - emotions_done) * rate
        print(f"  Time: {elapsed / 60:.1f}min elapsed, ~{remaining / 60:.1f}min remaining")

    # Summary
    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"Generation complete in {elapsed / 3600:.1f}h ({elapsed / 60:.1f}min)")
    print(f"  Generated: {total_generated}")
    print(f"  Skipped (already done): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Output: {output_dir}/stories/")
    print("=" * 70)

    if total_failed > 0:
        print(f"\nWARNING: {total_failed} stories failed to generate")


if __name__ == "__main__":
    main()
