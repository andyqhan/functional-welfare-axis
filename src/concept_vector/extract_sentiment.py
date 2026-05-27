"""
Extract sentiment concept vectors from the base Qwen3-4B model.

Uses difference-of-means on residual stream activations at target layers,
captured at the last token position (the \n after <|im_start|>assistant).

Dataset: Counterfactually Augmented Data (CAD) sentiment dataset.

Usage:
    uv run python -m src.concept_vector.extract_sentiment
    uv run python -m src.concept_vector.extract_sentiment --target-layers 20 21 22 23
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .activation_extraction import get_activations_pre_hook
from .hook_utils import add_hooks
from .model_utils import (
    DEFAULT_BASE_MODEL,
    detect_model_format_from_tokenizer,
    format_generation_prompt_harmony,
    get_model_block_modules,
    load_base_model,
)


PROMPT_TEMPLATE = "Text: {text}\n\nQuestion: Is the overall sentiment of the text positive or negative?"


def load_dataset(path: str) -> tuple[list[str], list[str]]:
    """Load CAD sentiment dataset and split by label."""
    df = pd.read_csv(path, sep="\t")
    assert {"Sentiment", "Text"}.issubset(df.columns), (
        f"Expected columns 'Sentiment' and 'Text', got {list(df.columns)}"
    )
    assert df["Text"].notna().all(), "Found null Text values"
    sentiment_values = set(df["Sentiment"].unique())
    assert sentiment_values == {"Positive", "Negative"}, (
        f"Expected exactly {{'Positive', 'Negative'}}, got {sentiment_values}"
    )

    positive_texts = df[df["Sentiment"] == "Positive"]["Text"].tolist()
    negative_texts = df[df["Sentiment"] == "Negative"]["Text"].tolist()
    print(f"Loaded {len(df)} samples: {len(positive_texts)} positive, {len(negative_texts)} negative")
    return positive_texts, negative_texts


def tokenize_sentiment_batch(texts: list[str], tokenizer, batch_size: int = 32):
    """Format texts as chat prompts and tokenize in one batch."""
    formatted = []
    for text in texts:
        prompt = PROMPT_TEMPLATE.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    # Spot-check: last token of first prompt should be \n (token 198)
    first_ids = tokenizer.encode(formatted[0], add_special_tokens=False)
    last_decoded = tokenizer.decode([first_ids[-1]]).strip()
    assert last_decoded == "", (
        f"Expected last token to be whitespace/newline, got token {first_ids[-1]} "
        f"({repr(tokenizer.decode([first_ids[-1]]))})"
    )

    # Batch tokenize with left-padding
    inputs = tokenizer(
        formatted,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    assert inputs.input_ids.shape[0] == len(texts)
    return inputs


def extract_sentiment_activations(
    model,
    tokenizer,
    texts: list[str],
    target_layers: list[int],
    batch_size: int = 32,
    model_format: str = "chatml",
) -> torch.Tensor:
    """
    Extract mean activations at position [-1] for target layers only.

    Returns:
        mean_cache: shape (1, len(target_layers), hidden_size) in float64
    """
    block_modules = get_model_block_modules(model)
    n_target = len(target_layers)
    d_model = model.config.hidden_size
    n_samples = len(texts)

    # Validate target layers are in range
    n_layers = model.config.num_hidden_layers
    for layer_idx in target_layers:
        assert 0 <= layer_idx < n_layers, (
            f"Layer {layer_idx} out of range [0, {n_layers})"
        )

    mean_cache = torch.zeros(
        (1, n_target, d_model),
        dtype=torch.float64,
        device=model.device,
    )

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting"):
            batch_texts = texts[i : i + batch_size]

            # Format and tokenize this batch
            formatted = []
            for text in batch_texts:
                prompt = PROMPT_TEMPLATE.format(text=text)
                messages = [{"role": "user", "content": prompt}]
                if model_format == "harmony":
                    formatted.append(format_generation_prompt_harmony(messages))
                else:
                    formatted.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    ))

            inputs = tokenizer(
                formatted, padding=True, truncation=False, return_tensors="pt",
                add_special_tokens=(model_format != "harmony"),
            )

            # Register hooks only on target layers, mapping to cache indices 0..n_target-1
            fwd_pre_hooks = [
                (
                    block_modules[layer_idx],
                    get_activations_pre_hook(
                        layer=cache_idx,
                        mean_cache=mean_cache,
                        sample_cache=None,
                        sample_offset=i,
                        n_samples=n_samples,
                        positions=[-1],
                    ),
                )
                for cache_idx, layer_idx in enumerate(target_layers)
            ]

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    assert not mean_cache.isnan().any(), "NaN values in mean activations"
    assert not mean_cache.isinf().any(), "Inf values in mean activations"
    return mean_cache


def main():
    parser = argparse.ArgumentParser(description="Extract sentiment concept vectors")
    parser.add_argument("--dataset", default="datasets/cad_sentiment_train.tsv")
    parser.add_argument("--output", default="artifacts/sentiment_vectors")
    parser.add_argument("--model", default=None, help="Base model path (default: auto from model_utils)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-layers", type=int, nargs="+", default=[20, 21, 22, 23])
    args = parser.parse_args()

    t0 = time.time()

    # 1. Load dataset
    positive_texts, negative_texts = load_dataset(args.dataset)

    # 2. Load model
    model, tokenizer = load_base_model(base_model_path=args.model)
    model_format = detect_model_format_from_tokenizer(tokenizer)
    print(f"Detected model format: {model_format}")

    # 3. Spot-check tokenization
    first_prompt = PROMPT_TEMPLATE.format(text=positive_texts[0])
    messages = [{"role": "user", "content": first_prompt}]
    if model_format == "harmony":
        first_formatted = format_generation_prompt_harmony(messages)
    else:
        first_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    first_ids = tokenizer.encode(first_formatted, add_special_tokens=False)
    last_decoded = tokenizer.decode([first_ids[-1]])
    if model_format == "harmony":
        assert last_decoded == "<|message|>", (
            f"Expected last token to be <|message|>, got token {first_ids[-1]} "
            f"({repr(last_decoded)})"
        )
    else:
        assert last_decoded.strip() == "", (
            f"Expected last token to be whitespace/newline, got token {first_ids[-1]} "
            f"({repr(last_decoded)})"
        )
    print(f"Spot-check passed: last token is {repr(last_decoded)} (id={first_ids[-1]})")
    print(f"Sample formatted prompt (first 200 chars): {first_formatted[:200]}")

    # 4. Extract positive activations
    print(f"\n=== Extracting POSITIVE activations ({len(positive_texts)} samples) ===")
    mean_positive = extract_sentiment_activations(
        model, tokenizer, positive_texts, args.target_layers, args.batch_size,
        model_format=model_format,
    )
    print(f"mean_positive shape: {mean_positive.shape}, norms: {mean_positive.norm(dim=-1)}")

    torch.cuda.empty_cache()

    # 5. Extract negative activations
    print(f"\n=== Extracting NEGATIVE activations ({len(negative_texts)} samples) ===")
    mean_negative = extract_sentiment_activations(
        model, tokenizer, negative_texts, args.target_layers, args.batch_size,
        model_format=model_format,
    )
    print(f"mean_negative shape: {mean_negative.shape}, norms: {mean_negative.norm(dim=-1)}")

    torch.cuda.empty_cache()

    # 6. Compute difference-of-means
    mean_diff = mean_positive - mean_negative
    assert mean_diff.shape == mean_positive.shape
    print(f"\nmean_diff norms per layer: {mean_diff.norm(dim=-1)}")

    # 7. Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(mean_positive.cpu(), output_dir / "mean_positive.pt")
    torch.save(mean_negative.cpu(), output_dir / "mean_negative.pt")
    torch.save(mean_diff.cpu(), output_dir / "mean_diff.pt")

    elapsed = time.time() - t0
    metadata = {
        "n_positive": len(positive_texts),
        "n_negative": len(negative_texts),
        "target_layers": args.target_layers,
        "model": args.model or DEFAULT_BASE_MODEL,
        "prompt_template": PROMPT_TEMPLATE,
        "position": [-1],
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "output_shapes": {
            "mean_positive": list(mean_positive.shape),
            "mean_negative": list(mean_negative.shape),
            "mean_diff": list(mean_diff.shape),
        },
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  mean_positive.pt: {mean_positive.shape}")
    print(f"  mean_negative.pt: {mean_negative.shape}")
    print(f"  mean_diff.pt:     {mean_diff.shape}")
    print(f"  metadata.json")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
