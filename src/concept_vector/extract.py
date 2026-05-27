"""
CLI entry point for concept vector extraction.

Extracts concept vectors for a specified concept type (lava, goal, or path).
The concept vector is computed as: mean(positive_class) - mean(negative_class)

Concept types:
  - lava: LAVA vs (PATH + GOAL) - detects "lava" concept
  - goal: GOAL vs (PATH + LAVA) - detects "goal" concept
  - path: PATH vs (GOAL + LAVA) - detects "path" concept

Usage:
    python -m src.concept_vector.extract \
        --checkpoint runs/maze-grpo-128k-706120/global_step_175/actor \
        --output artifacts/concept_vectors/step_175

    # Extract goal concept vector
    python -m src.concept_vector.extract \
        --checkpoint runs/maze-grpo-128k-706120/global_step_175/actor \
        --concept goal \
        --output artifacts/concept_vectors/step_175/goal

    # Extract from base model only (no LoRA adapter)
    python -m src.concept_vector.extract \
        --base-model-only \
        --concept lava \
        --output artifacts/concept_vectors/baseline/lava

    # With options
    python -m src.concept_vector.extract \
        --checkpoint runs/maze-grpo-128k-706120/global_step_175/actor \
        --dataset datasets/concept_vector_big.parquet \
        --batch-size 64 \
        --output artifacts/concept_vectors/step_175
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from .model_utils import load_model_with_lora, load_base_model, resolve_base_model_path, detect_use_chat_template, DEFAULT_BASE_MODEL
from .activation_extraction import get_mean_diff


def generate_output_dir_name(checkpoint_path: Path) -> str:
    """
    Generate a descriptive output directory name from checkpoint path.

    Examples:
        runs/maze-grpo-128k-706120/global_step_175/actor
            → maze-grpo-128k-706120_step175
        runs/maze-grpo-128k-continuous-wind-big-warmup-ka8192-1136714/global_step_205/actor
            → maze-grpo-128k-ka8192-1136714_step205

    The naming includes the training run identifier and step number for traceability.
    """
    # Navigate up from checkpoint path to find run dir and step dir
    # Typical structure: runs/<run_name>/global_step_N/actor
    if checkpoint_path.name == "actor":
        step_dir = checkpoint_path.parent.name  # e.g., "global_step_205"
        run_dir = checkpoint_path.parent.parent.name  # e.g., "maze-grpo-128k-..."
    else:
        step_dir = checkpoint_path.name
        run_dir = checkpoint_path.parent.name

    # Extract step number from step_dir (e.g., "global_step_205" → "205")
    step_num = step_dir.replace("global_step_", "").replace("step_", "").replace("step", "")

    # Abbreviate run_dir if it's too long (keep job ID at the end)
    # Run names typically end with a job ID (e.g., "-1136714")
    if len(run_dir) > 40:
        # Try to extract job ID (last numeric segment after a hyphen)
        parts = run_dir.split("-")
        # Find last part that looks like a job ID (all digits, reasonable length)
        job_id = None
        for part in reversed(parts):
            if part.isdigit() and len(part) >= 5:
                job_id = part
                break

        if job_id:
            # Keep prefix and job ID
            prefix_parts = []
            for part in parts:
                if part == job_id:
                    break
                prefix_parts.append(part)
                # Stop after a few meaningful parts
                if len("-".join(prefix_parts)) > 20:
                    break
            run_dir = "-".join(prefix_parts[:3]) + "-" + job_id

    return f"{run_dir}_step{step_num}"


VALID_CONCEPTS = {"lava", "goal", "path"}


def main(
    checkpoint_path: str,
    dataset_path: str = "datasets/concept_vector_big.parquet",
    output_dir: str = None,
    batch_size: int = 32,
    positions: list = None,
    store_individual: bool = False,
    concept: str = "lava",
    base_model_only: bool = False,
    base_model_path: str = None,
    no_chat_template: bool = False,
):
    """
    Extract concept vectors from maze trajectories.

    Computes the activation difference between trajectories ending on the
    target concept tile vs. other tiles, saving the result as a concept vector.

    Args:
        checkpoint_path: Path to checkpoint (e.g., runs/maze-grpo-xxx/global_step_N/actor)
        dataset_path: Path to concept_vector_big.parquet
        output_dir: Where to save mean_diff.pt and metadata.json
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1])
        store_individual: If True, also save per-sample activations for separability analysis
        concept: Which concept to extract: "lava", "goal", or "path"
        base_model_only: If True, load base model without LoRA adapter
        base_model_path: Override base model path (default: Qwen3-4B-Instruct-2507)
    """
    if positions is None:
        positions = [-1]

    # FAIL FAST: Validate concept
    assert concept in VALID_CONCEPTS, f"Invalid concept: {concept}. Must be one of {VALID_CONCEPTS}"

    # Determine model source
    if base_model_only:
        model_source = base_model_path or DEFAULT_BASE_MODEL
        checkpoint_path = None
    else:
        checkpoint_path = Path(checkpoint_path)
        # FAIL FAST: Validate checkpoint path
        assert checkpoint_path.exists(), f"Checkpoint path does not exist: {checkpoint_path}"
        model_source = str(checkpoint_path)

    # Auto-generate output directory if not specified
    if output_dir is None:
        if base_model_only:
            # For base model, use "baseline" as the directory name
            output_dir = Path("artifacts/concept_vectors") / "baseline"
        else:
            dir_name = generate_output_dir_name(checkpoint_path)
            output_dir = Path("artifacts/concept_vectors") / dir_name
    else:
        output_dir = Path(output_dir)

    print(f"=== Concept Vector Extraction ===")
    print(f"Concept: {concept.upper()}")
    print(f"Model: {'BASE MODEL (no LoRA)' if base_model_only else f'LoRA checkpoint: {checkpoint_path}'}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Positions: {positions}")
    print(f"Store individual: {store_individual}")

    # 1. Load model
    print(f"\n--- Loading model ---")
    if base_model_only:
        model, tokenizer = load_base_model(base_model_path)
        use_chat_template = not no_chat_template  # default True; --no-chat-template forces False
    else:
        model, tokenizer = load_model_with_lora(str(checkpoint_path))
        use_chat_template = False if no_chat_template else detect_use_chat_template(str(checkpoint_path))

    format_kwargs = {} if use_chat_template else {"use_chat_template": False}
    if not use_chat_template:
        print(f"  Detected plain-text format (use_chat_template=False)")

    # 2. Load dataset
    print(f"\n--- Loading dataset ---")
    df = pd.read_parquet(dataset_path)

    # FAIL FAST: Validate dataset structure
    assert 'final_tile' in df.columns, "Dataset missing 'final_tile' column"
    assert 'trajectory' in df.columns, "Dataset missing 'trajectory' column"

    unique_tiles = set(df['final_tile'].unique())
    expected_tiles = {'LAVA', 'PATH', 'GOAL'}
    assert unique_tiles <= expected_tiles, (
        f"Unexpected final_tile values: {unique_tiles - expected_tiles}"
    )

    # Split by final tile based on concept type
    # positive_class = the concept we're detecting
    # negative_class = everything else
    concept_upper = concept.upper()
    positive_df = df[df['final_tile'] == concept_upper]
    negative_df = df[df['final_tile'] != concept_upper]

    # Convert trajectories to lists (parquet stores them as numpy arrays)
    positive_trajs = [list(traj) for traj in positive_df['trajectory'].tolist()]
    negative_trajs = [list(traj) for traj in negative_df['trajectory'].tolist()]

    # Count samples per tile type for logging
    n_lava = len(df[df['final_tile'] == 'LAVA'])
    n_path = len(df[df['final_tile'] == 'PATH'])
    n_goal = len(df[df['final_tile'] == 'GOAL'])

    print(f"Dataset breakdown: LAVA={n_lava}, PATH={n_path}, GOAL={n_goal}")
    print(f"Concept '{concept}': positive={len(positive_trajs)} ({concept_upper}), negative={len(negative_trajs)} (other)")

    # FAIL FAST: Validate we have samples
    assert len(positive_trajs) > 0, f"No {concept_upper} trajectories in dataset"
    assert len(negative_trajs) > 0, f"No non-{concept_upper} trajectories in dataset"

    # 3. Extract mean activations and compute concept vector
    # Note: get_mean_diff expects (positive_class, negative_class) trajectories
    # The function internally names them "lava" and "other" but we're using it generically
    mean_diff, mean_positive, mean_negative, activations_positive, activations_negative = get_mean_diff(
        model=model,
        tokenizer=tokenizer,
        lava_trajectories=positive_trajs,  # positive class (concept we're detecting)
        other_trajectories=negative_trajs,  # negative class (everything else)
        batch_size=batch_size,
        positions=positions,
        store_individual=store_individual,
        format_kwargs=format_kwargs,
    )

    # FAIL FAST: Final validation
    expected_shape = (len(positions), model.config.num_hidden_layers, model.config.hidden_size)
    assert mean_diff.shape == expected_shape, (
        f"Output shape {mean_diff.shape} != expected {expected_shape}"
    )
    assert not mean_diff.isnan().any(), "NaN in final mean_diff"

    # 4. Save outputs
    print(f"\n--- Saving outputs ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mean tensors
    torch.save(mean_diff, output_dir / "mean_diff.pt")
    torch.save(mean_positive, output_dir / "mean_lava.pt")  # positive class mean
    torch.save(mean_negative, output_dir / "mean_other.pt")  # negative class mean

    # Save individual activations if requested
    # Use "activations_lava" and "activations_other" for compatibility with precompute.py
    if store_individual and activations_positive is not None:
        torch.save(activations_positive, output_dir / "activations_lava.pt")
        torch.save(activations_negative, output_dir / "activations_other.pt")
        print(f"  Individual activations saved:")
        print(f"    activations_lava.pt (positive/{concept}): {activations_positive.shape}")
        print(f"    activations_other.pt (negative/other): {activations_negative.shape}")

    # Save metadata
    used_base_model = resolve_base_model_path(str(checkpoint_path), base_model_path) if checkpoint_path else (base_model_path or DEFAULT_BASE_MODEL)
    metadata = {
        "concept": concept,
        "positive_class": concept_upper,
        "negative_class": [t for t in ['LAVA', 'PATH', 'GOAL'] if t != concept_upper],
        "base_model_only": base_model_only,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "base_model": used_base_model,
        "dataset_path": str(dataset_path),
        "n_positive": len(positive_trajs),
        "n_negative": len(negative_trajs),
        "n_lava": n_lava,
        "n_path": n_path,
        "n_goal": n_goal,
        "positions": positions,
        "extraction_position": "last_token_of_final_move",
        "batch_size": batch_size,
        "store_individual": store_individual,
        "tensor_shape": list(mean_diff.shape),
        "tensor_dtype": str(mean_diff.dtype),
        "timestamp": datetime.now().isoformat(),
    }

    if store_individual and activations_positive is not None:
        metadata["activations_positive_shape"] = list(activations_positive.shape)
        metadata["activations_negative_shape"] = list(activations_negative.shape)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved:")
    print(f"  {output_dir / 'mean_diff.pt'}")
    print(f"  {output_dir / 'mean_lava.pt'} (positive class: {concept_upper})")
    print(f"  {output_dir / 'mean_other.pt'} (negative class)")
    if store_individual and activations_positive is not None:
        print(f"  {output_dir / 'activations_lava.pt'} (positive class)")
        print(f"  {output_dir / 'activations_other.pt'} (negative class)")
    print(f"  {output_dir / 'metadata.json'}")

    # Report final statistics
    print(f"\n=== Extraction complete ===")
    print(f"Concept: {concept}")
    print(f"Concept vector shape: {mean_diff.shape}")
    diff_norms = mean_diff.norm(dim=-1).squeeze()  # (n_layers,)
    print(f"Layer norm range: [{diff_norms.min():.4f}, {diff_norms.max():.4f}]")
    print(f"Top 5 layers by norm: {diff_norms.argsort(descending=True)[:5].tolist()}")

    return mean_diff, mean_positive, mean_negative, activations_positive, activations_negative


def cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Extract concept vectors from maze trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g., runs/maze-grpo-xxx/global_step_N/actor). Required unless --base-model-only is specified.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/concept_vector_big.parquet",
        help="Path to concept_vector_big.parquet dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for forward passes",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[-1],
        help="Token positions to extract activations from",
    )
    parser.add_argument(
        "--no-store-individual",
        action="store_true",
        help="Skip saving per-sample activations (saves disk space)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="lava",
        choices=["lava", "goal", "path"],
        help="Concept type to extract: lava, goal, or path (default: lava)",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Load base model without LoRA adapter (for baseline comparison)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Force use_chat_template=False (for base model runs trained without chat format)",
    )

    args = parser.parse_args()

    # Validate: either checkpoint or base-model-only must be specified
    if not args.base_model_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --base-model-only is specified")

    main(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        positions=args.positions,
        store_individual=not args.no_store_individual,
        concept=args.concept,
        base_model_only=args.base_model_only,
        base_model_path=args.base_model,
        no_chat_template=args.no_chat_template,
    )


if __name__ == "__main__":
    cli()
