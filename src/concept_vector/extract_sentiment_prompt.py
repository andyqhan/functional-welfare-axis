"""
Extract sentiment concept vector from two contrasting prompts.

Computes difference-of-activations between:
  "Describe a book using positive sentiment"
  "Describe a book using negative sentiment"

Usage:
    uv run python -m src.concept_vector.extract_sentiment_prompt
"""

import json
import time
from pathlib import Path

import torch

from .activation_extraction import get_activations_pre_hook
from .hook_utils import add_hooks
from .model_utils import (
    DEFAULT_BASE_MODEL,
    detect_model_format_from_tokenizer,
    format_generation_prompt_harmony,
    get_model_block_modules,
    load_base_model,
)


POSITIVE_PROMPT = "Describe a book using positive sentiment"
NEGATIVE_PROMPT = "Describe a book using negative sentiment"
TARGET_LAYERS = [20, 21, 22, 23]


def extract_single_prompt(model, tokenizer, prompt_text, target_layers, model_format="chatml"):
    """Run a single prompt through the model and return activations at target layers."""
    block_modules = get_model_block_modules(model)
    n_target = len(target_layers)
    d_model = model.config.hidden_size

    mean_cache = torch.zeros(
        (1, n_target, d_model), dtype=torch.float64, device=model.device,
    )

    messages = [{"role": "user", "content": prompt_text}]
    if model_format == "harmony":
        formatted = format_generation_prompt_harmony(messages)
    else:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    inputs = tokenizer(
        formatted, return_tensors="pt",
        add_special_tokens=(model_format != "harmony"),
    )

    # Verify last token matches expected format
    last_tok = inputs.input_ids[0, -1].item()
    last_decoded = tokenizer.decode([last_tok])
    if model_format == "harmony":
        assert last_decoded == "<|message|>", (
            f"Expected last token to be <|message|>, got token {last_tok} ({repr(last_decoded)})"
        )
    else:
        assert last_decoded.strip() == "", (
            f"Expected last token to be whitespace/newline, got token {last_tok} ({repr(last_decoded)})"
        )

    fwd_pre_hooks = [
        (
            block_modules[layer_idx],
            get_activations_pre_hook(
                layer=cache_idx,
                mean_cache=mean_cache,
                sample_cache=None,
                sample_offset=0,
                n_samples=1,
                positions=[-1],
            ),
        )
        for cache_idx, layer_idx in enumerate(target_layers)
    ]

    with torch.no_grad():
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    assert not mean_cache.isnan().any()
    return mean_cache


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract sentiment vector from 2 contrasting prompts")
    parser.add_argument("--model", default=None, help="Base model path (default: auto from model_utils)")
    parser.add_argument("--output", default="artifacts/sentiment_vectors_prompt")
    parser.add_argument("--target-layers", type=int, nargs="+", default=TARGET_LAYERS)
    args = parser.parse_args()

    t0 = time.time()

    model, tokenizer = load_base_model(base_model_path=args.model)
    model_format = detect_model_format_from_tokenizer(tokenizer)
    print(f"Detected model format: {model_format}")

    print(f"\nPositive prompt: {POSITIVE_PROMPT!r}")
    act_positive = extract_single_prompt(model, tokenizer, POSITIVE_PROMPT, args.target_layers, model_format)
    print(f"  norms: {act_positive.norm(dim=-1)}")

    print(f"\nNegative prompt: {NEGATIVE_PROMPT!r}")
    act_negative = extract_single_prompt(model, tokenizer, NEGATIVE_PROMPT, args.target_layers, model_format)
    print(f"  norms: {act_negative.norm(dim=-1)}")

    mean_diff = act_positive - act_negative
    print(f"\nmean_diff norms per layer: {mean_diff.norm(dim=-1)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(act_positive.cpu(), output_dir / "mean_positive.pt")
    torch.save(act_negative.cpu(), output_dir / "mean_negative.pt")
    torch.save(mean_diff.cpu(), output_dir / "mean_diff.pt")

    elapsed = time.time() - t0
    metadata = {
        "positive_prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "target_layers": args.target_layers,
        "model": args.model or DEFAULT_BASE_MODEL,
        "position": [-1],
        "output_shapes": list(mean_diff.shape),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
