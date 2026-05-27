"""
Logit lens analysis for concept vectors.

Multiplies concept vectors by the model's unembedding matrix (lm_head.weight)
to reveal which vocabulary tokens each layer's concept direction promotes or
suppresses. No generation or forward passes required — just a matmul.

Usage:
    # Single concept directory
    python -m src.concept_vector.logit_lens \
        --dir artifacts/concept_vectors/run_name/lava

    # All concepts in an extraction directory
    python -m src.concept_vector.logit_lens \
        --dir artifacts/concept_vectors/run_name

    # Custom top-k
    python -m src.concept_vector.logit_lens \
        --dir artifacts/concept_vectors/run_name --top-k 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


def _compute_logit_lens_for_vector(
    concept_vector: torch.Tensor,
    W_unembed: torch.Tensor,
    tokenizer,
    top_k: int = 20,
) -> dict:
    """Project concept vector through unembedding matrix for all layers.

    Args:
        concept_vector: Shape (n_layers, d_model), float32.
        W_unembed: Shape (vocab_size, d_model), float32.
        tokenizer: For decoding token IDs to strings.
        top_k: Number of top/bottom tokens per layer.

    Returns:
        Dict keyed by layer index (str) with promoted/suppressed tokens.
    """
    n_layers, d_model = concept_vector.shape
    assert W_unembed.shape[1] == d_model, (
        f"Dimension mismatch: concept_vector d_model={d_model}, "
        f"W_unembed d_model={W_unembed.shape[1]}"
    )
    assert top_k > 0, f"top_k must be positive, got {top_k}"

    results = {}
    for layer in range(n_layers):
        direction = concept_vector[layer]  # (d_model,)
        logits = W_unembed @ direction  # (vocab_size,)

        top_vals, top_ids = logits.topk(top_k)
        bot_vals, bot_ids = logits.topk(top_k, largest=False)

        promoted = [
            {
                "token": tokenizer.decode([tid.item()]),
                "token_id": tid.item(),
                "logit": round(val.item(), 6),
            }
            for val, tid in zip(top_vals, top_ids)
        ]
        suppressed = [
            {
                "token": tokenizer.decode([tid.item()]),
                "token_id": tid.item(),
                "logit": round(val.item(), 6),
            }
            for val, tid in zip(bot_vals, bot_ids)
        ]

        results[str(layer)] = {
            "promoted": promoted,
            "suppressed": suppressed,
            "direction_norm": round(direction.norm().item(), 6),
        }

    return results


def logit_lens_concept_vector(
    concept_dir: str | Path,
    W_unembed: torch.Tensor,
    tokenizer,
    top_k: int = 20,
    position: int = 0,
    output_path: str | Path | None = None,
) -> dict:
    """Compute logit lens analysis for a single concept vector directory.

    Args:
        concept_dir: Path to concept directory containing mean_diff.pt and metadata.json.
        W_unembed: Unembedding matrix, shape (vocab_size, d_model).
        tokenizer: For decoding token IDs.
        top_k: Number of top/bottom tokens per layer.
        position: Position index into mean_diff's first dimension.
        output_path: Where to save results JSON (default: concept_dir/logit_lens.json).

    Returns:
        Full results dict with metadata and per-layer results.
    """
    concept_dir = Path(concept_dir)
    mean_diff_path = concept_dir / "mean_diff.pt"
    assert mean_diff_path.exists(), f"mean_diff.pt not found at {mean_diff_path}"

    mean_diff = torch.load(mean_diff_path, map_location="cpu", weights_only=True)
    assert mean_diff.ndim == 3, f"Expected 3D tensor, got shape {mean_diff.shape}"
    assert position < mean_diff.shape[0], (
        f"Position {position} out of range for tensor with {mean_diff.shape[0]} positions"
    )

    # Extract concept vector for this position, cast to float32
    cv = mean_diff[position].float()  # (n_layers, d_model)

    # Load metadata if available
    metadata_path = concept_dir / "metadata.json"
    concept_name = concept_dir.name
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        concept_name = metadata.get("concept", concept_name)

    # Compute logit lens
    layers = _compute_logit_lens_for_vector(cv, W_unembed, tokenizer, top_k)

    result = {
        "metadata": {
            "concept": concept_name,
            "concept_vector_dir": str(concept_dir),
            "concept_vector_shape": list(mean_diff.shape),
            "position": position,
            "top_k": top_k,
            "vocab_size": W_unembed.shape[0],
            "timestamp": datetime.now().isoformat(),
        },
        "layers": layers,
    }

    # Save
    if output_path is None:
        output_path = concept_dir / "logit_lens.json"
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved logit lens for '{concept_name}' ({len(layers)} layers) to {output_path}")

    return result


def logit_lens_all_concepts(
    extraction_dir: str | Path,
    top_k: int = 20,
    base_model_path: str | None = None,
    position: int = 0,
    load_dtype: torch.dtype = torch.float32,
) -> dict[str, dict]:
    """Compute logit lens for all concepts in an extraction directory.

    Loads the model and tokenizer once, then processes all concepts.

    Args:
        extraction_dir: Top-level extraction directory or single concept directory.
        top_k: Number of top/bottom tokens per layer.
        base_model_path: Override base model path.
        position: Position index into mean_diff.
        load_dtype: Model load precision. Extracted unembedding is always
            upcast to float32 for the matmul, so this only affects peak RAM.

    Returns:
        Dict mapping concept name to results.
    """
    extraction_dir = Path(extraction_dir)

    # Discover concept directories
    concept_dirs = _discover_concepts(extraction_dir)
    assert len(concept_dirs) > 0, f"No concept vectors found in {extraction_dir}"
    print(f"Found {len(concept_dirs)} concept(s): {list(concept_dirs.keys())}")

    # Resolve base model path from first concept's metadata
    if base_model_path is None:
        base_model_path = _resolve_base_model(concept_dirs)

    # Load model for unembedding matrix
    W_unembed, tokenizer = _load_unembedding(base_model_path, load_dtype=load_dtype)

    # Process each concept
    results = {}
    for label, concept_dir in concept_dirs.items():
        print(f"\nProcessing concept: {label}")
        result = logit_lens_concept_vector(
            concept_dir, W_unembed, tokenizer,
            top_k=top_k, position=position,
        )
        results[label] = result

    return results


def _discover_concepts(extraction_dir: Path) -> dict[str, Path]:
    """Find concept vector directories. Returns {label: path}."""
    # Case 1: extraction_dir itself is a concept dir
    if (extraction_dir / "mean_diff.pt").exists():
        return {extraction_dir.name: extraction_dir}

    # Case 2: concepts.txt lists concept subdirs
    concepts_txt = extraction_dir / "concepts.txt"
    if concepts_txt.exists():
        with open(concepts_txt) as f:
            concepts = [line.strip() for line in f if line.strip()]
        result = {}
        for concept in concepts:
            concept_path = extraction_dir / concept
            if (concept_path / "mean_diff.pt").exists():
                result[concept] = concept_path
        if result:
            return result

    # Case 3: recursive search for mean_diff.pt
    result = {}
    for mean_diff_path in sorted(extraction_dir.rglob("mean_diff.pt")):
        rel = mean_diff_path.parent.relative_to(extraction_dir)
        label = str(rel)
        result[label] = mean_diff_path.parent

    return result


def _resolve_base_model(concept_dirs: dict[str, Path]) -> str:
    """Try to find base_model from metadata.json of the first concept."""
    from .model_utils import DEFAULT_BASE_MODEL

    for label, concept_dir in concept_dirs.items():
        metadata_path = concept_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            base_model = metadata.get("base_model")
            if base_model:
                print(f"Using base model from {label}/metadata.json: {base_model}")
                return base_model

    print(f"No base_model in metadata, using default: {DEFAULT_BASE_MODEL}")
    return DEFAULT_BASE_MODEL


def _load_unembedding(
    base_model_path: str, load_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, object]:
    """Load model to extract unembedding matrix, then free model memory.

    `load_dtype` controls the precision used to materialise the model on CPU
    (peak RAM is dominated by this). The returned unembedding matrix is
    always upcast to float32 so the downstream matmul is identical
    regardless of how the model was loaded. For large models (e.g.
    GPT-OSS-20B, ~84 GB in fp32) use bfloat16 to halve peak memory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model (dtype={load_dtype}) "
          f"for unembedding matrix: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=load_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Always return float32 — cheap (a few GB for the unembedding alone)
    # and keeps the matmul bit-identical across load_dtype choices.
    W_unembed = model.lm_head.weight.detach().float().clone()
    print(f"Unembedding matrix shape: {W_unembed.shape}, dtype=float32")
    del model

    return W_unembed, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for concept vectors",
    )
    parser.add_argument(
        "--dir", required=True, type=str,
        help="Concept vector directory (single concept or extraction root)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top/bottom tokens per layer (default: 20)",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model path (default: from metadata.json or Qwen3-4B)",
    )
    parser.add_argument(
        "--position", type=int, default=0,
        help="Position index into concept vector (default: 0)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "bfloat16", "float16"],
        help=(
            "Model load dtype (default: float32). The extracted unembedding "
            "is always upcast to float32 for the matmul, so this only "
            "controls peak RAM during model load. Use bfloat16 to fit large "
            "models (e.g. GPT-OSS-20B) in less memory."
        ),
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    logit_lens_all_concepts(
        extraction_dir=args.dir,
        top_k=args.top_k,
        base_model_path=args.base_model,
        position=args.position,
        load_dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
