"""
Extract emoji concept vectors from Qwen3-4B-Instruct-2507.

For each Unicode emoji, prompts the model with "Tell me about {emoji}", extracts
the activation at the generation prompt position (last token) across all layers,
then computes leave-one-out concept vectors and measures cosine similarity with
pre-extracted sentiment concept vectors.

Usage:
    python -m src.concept_vector.extract_emoji_vectors \
        --output artifacts/emoji_vectors

    python -m src.concept_vector.extract_emoji_vectors \
        --output artifacts/emoji_vectors --batch-size 64
"""

import argparse
import json
import time
import os
from pathlib import Path

import emoji
import torch
import torch.nn.functional as F

from .model_utils import load_base_model, format_trajectory, DEFAULT_BASE_MODEL
from .activation_extraction import get_activations


def compute_leave_one_out_vectors(activations: torch.Tensor) -> torch.Tensor:
    """Compute leave-one-out concept vectors.

    For each item i: concept_vector[i] = activation[i] - mean(activations[j != i])
    """
    N = activations.shape[0]
    assert N > 1, f"Need at least 2 items, got {N}"
    act_f64 = activations.to(torch.float64)
    total_sum = act_f64.sum(dim=0)
    concept_vectors = torch.zeros_like(activations)
    for i in range(N):
        others_mean = (total_sum - act_f64[i]) / (N - 1)
        concept_vectors[i] = (act_f64[i] - others_mean).to(torch.float32)
    return concept_vectors


def get_all_emoji() -> list[tuple[str, str]]:
    """
    Get all Unicode emoji with their English names.

    Returns:
        List of (emoji_char, english_name) tuples, sorted by emoji char for reproducibility.
    """
    results = []
    for e, data in emoji.EMOJI_DATA.items():
        en_name = data.get("en", "")
        # Strip surrounding colons from name (e.g. ":grinning_face:" -> "grinning_face")
        en_name = en_name.strip(":")
        results.append((e, en_name))

    # Sort by emoji character for reproducibility
    results.sort(key=lambda x: x[0])

    assert len(results) > 1000, f"Expected >1000 emoji, got {len(results)}"
    return results


def create_trajectories(emoji_list: list[str]) -> list[list[dict]]:
    """Create chat trajectories for each emoji: 'Tell me about {emoji}'."""
    trajectories = []
    for e in emoji_list:
        traj = [
            {"role": "user", "content": f"Tell me about {e}"},
            {"role": "assistant", "content": ""},
        ]
        trajectories.append(traj)
    return trajectories


def load_sentiment_vectors(
    cad_dir: str = "artifacts/sentiment_vectors",
    prompt_dir: str = "artifacts/sentiment_vectors_prompt",
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """
    Load pre-extracted sentiment concept vectors.

    Returns:
        (mean_diff_cad, mean_diff_prompt, metadata_cad, metadata_prompt)
    """
    cad_path = Path(cad_dir)
    prompt_path = Path(prompt_dir)

    assert cad_path.exists(), f"CAD sentiment vectors not found: {cad_path}"
    assert prompt_path.exists(), f"Prompt sentiment vectors not found: {prompt_path}"

    mean_diff_cad = torch.load(cad_path / "mean_diff.pt", weights_only=True)
    mean_diff_prompt = torch.load(prompt_path / "mean_diff.pt", weights_only=True)

    with open(cad_path / "metadata.json") as f:
        meta_cad = json.load(f)
    with open(prompt_path / "metadata.json") as f:
        meta_prompt = json.load(f)

    # Validate shapes (1 position, N target layers, hidden_size)
    assert mean_diff_cad.ndim == 3 and mean_diff_cad.shape[0] == 1, f"Unexpected CAD shape: {mean_diff_cad.shape}"
    assert mean_diff_prompt.ndim == 3 and mean_diff_prompt.shape[0] == 1, f"Unexpected prompt shape: {mean_diff_prompt.shape}"

    # Validate both methods have the same last target layer
    assert meta_cad["target_layers"][-1] == meta_prompt["target_layers"][-1], (
        f"CAD and prompt target_layers mismatch: {meta_cad['target_layers']} vs {meta_prompt['target_layers']}"
    )

    print(f"Loaded CAD sentiment vector: {mean_diff_cad.shape}")
    print(f"Loaded prompt sentiment vector: {mean_diff_prompt.shape}")
    print(f"Target layers (CAD): {meta_cad['target_layers']}")
    print(f"Target layers (prompt): {meta_prompt['target_layers']}")

    return mean_diff_cad, mean_diff_prompt, meta_cad, meta_prompt


def extract_emoji_activations(
    model,
    tokenizer,
    emoji_list: list[str],
    *,
    batch_size: int = 64,
    inject_no_think: bool = False,
) -> torch.Tensor:
    """Extract activations for a specific list of emoji.

    Prompts the model with "Tell me about {emoji}" for each emoji in the list,
    extracts activation at position[-1] across all layers.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        emoji_list: List of emoji characters to extract activations for
        batch_size: Batch size for forward passes
        inject_no_think: If True, inject <think>\\n\\n</think>\\n\\n tokens

    Returns:
        Tensor of shape (N, 1, n_layers, d_model) with per-emoji activations
    """
    assert len(emoji_list) > 0, "Empty emoji list"

    trajectories = create_trajectories(emoji_list)
    format_kwargs = {"inject_no_think": True} if inject_no_think else {}

    _, individual_activations = get_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories=trajectories,
        batch_size=batch_size,
        positions=[-1],
        store_individual=True,
        format_kwargs=format_kwargs if format_kwargs else None,
    )

    assert individual_activations is not None
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    assert individual_activations.shape == (len(emoji_list), 1, n_layers, d_model), (
        f"Shape mismatch: {individual_activations.shape} != "
        f"({len(emoji_list)}, 1, {n_layers}, {d_model})"
    )
    assert not individual_activations.isnan().any(), "NaN in activations"

    return individual_activations


def main():
    parser = argparse.ArgumentParser(description="Extract emoji concept vectors")
    parser.add_argument("--output", type=str, default="artifacts/emoji_vectors",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for activation extraction")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model path (default: Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--sentiment-cad-dir", type=str, default="artifacts/sentiment_vectors",
                        help="Directory with CAD sentiment vectors")
    parser.add_argument("--sentiment-prompt-dir", type=str, default="artifacts/sentiment_vectors_prompt",
                        help="Directory with prompt-based sentiment vectors")
    parser.add_argument("--inject-no-think", action="store_true",
                        help="Inject <think>\\n\\n</think>\\n\\n after assistant header (for thinking models like Qwen3-8B)")
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output)

    print("=" * 60)
    print("Emoji Concept Vector Extraction")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Base model: {args.base_model or 'default'}")
    print(f"  Inject no-think: {args.inject_no_think}")
    print()

    # Step 1: Get all emoji
    print("=== Step 1: Getting all emoji ===")
    emoji_data = get_all_emoji()
    emoji_chars = [e for e, _ in emoji_data]
    emoji_names = [n for _, n in emoji_data]
    n_emoji = len(emoji_chars)
    print(f"Total emoji: {n_emoji}")
    print(f"First 5: {emoji_chars[:5]} ({emoji_names[:5]})")
    print(f"Last 5: {emoji_chars[-5:]} ({emoji_names[-5:]})")
    print()

    # Step 2: Create trajectories
    print("=== Step 2: Creating trajectories ===")
    trajectories = create_trajectories(emoji_chars)
    assert len(trajectories) == n_emoji

    # Verify format_trajectory produces expected last token
    format_kwargs = {"inject_no_think": True} if args.inject_no_think else {}
    sample_formatted = format_trajectory(trajectories[0], **format_kwargs)
    print(f"Sample trajectory for '{emoji_chars[0]}':")
    print(f"  {repr(sample_formatted[:120])}...")
    assert "<|im_start|>user" in sample_formatted
    assert "<|im_start|>assistant" in sample_formatted
    if args.inject_no_think:
        assert "<think>\n\n</think>\n\n" in sample_formatted, "inject_no_think failed: <think> block not found"
    print()

    # Step 3: Load model
    print("=== Step 3: Loading model ===")
    model, tokenizer = load_base_model(base_model_path=args.base_model)

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden size: {d_model}")

    # Verify last token: \n (token 198) normally, \n\n (token 271) with inject_no_think
    sample_tokens = tokenizer(sample_formatted, return_tensors="pt").input_ids[0]
    expected_last_token = 271 if args.inject_no_think else 198
    assert sample_tokens[-1].item() == expected_last_token, (
        f"Expected last token {expected_last_token}, got {sample_tokens[-1].item()} "
        f"({repr(tokenizer.decode(sample_tokens[-1]))})"
    )
    print(f"  Last token verification: {sample_tokens[-1].item()} = {repr(tokenizer.decode(sample_tokens[-1]))}")
    print()

    # Step 4: Extract activations
    print("=== Step 4: Extracting activations ===")
    extraction_start = time.time()

    mean_activations, individual_activations = get_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories=trajectories,
        batch_size=args.batch_size,
        positions=[-1],
        store_individual=True,
        format_kwargs=format_kwargs if format_kwargs else None,
    )

    extraction_time = time.time() - extraction_start
    print(f"Extraction took {extraction_time:.1f}s")

    assert individual_activations is not None
    assert individual_activations.shape == (n_emoji, 1, n_layers, d_model), (
        f"Shape mismatch: {individual_activations.shape} != ({n_emoji}, 1, {n_layers}, {d_model})"
    )
    assert not individual_activations.isnan().any(), "NaN in individual activations"
    print(f"Individual activations shape: {individual_activations.shape}")
    print()

    # Step 5: Compute leave-one-out concept vectors
    print("=== Step 5: Computing leave-one-out concept vectors ===")
    vector_start = time.time()
    concept_vectors = compute_leave_one_out_vectors(individual_activations)
    vector_time = time.time() - vector_start
    print(f"Concept vector computation took {vector_time:.1f}s")
    print(f"Concept vectors shape: {concept_vectors.shape}")
    assert not concept_vectors.isnan().any(), "NaN in concept vectors"
    print()

    # Step 6: Load sentiment vectors and compute cosine similarity
    print("=== Step 6: Loading sentiment vectors ===")
    mean_diff_cad, mean_diff_prompt, meta_cad, meta_prompt = load_sentiment_vectors(
        cad_dir=args.sentiment_cad_dir,
        prompt_dir=args.sentiment_prompt_dir,
    )

    # Extract the last target layer from sentiment vectors
    sent_target_layer = meta_cad["target_layers"][-1]
    sent_layer_idx = len(meta_cad["target_layers"]) - 1
    sent_cad = mean_diff_cad[0, sent_layer_idx, :]  # (d_model,)
    sent_prompt = mean_diff_prompt[0, sent_layer_idx, :]  # (d_model,)

    # Emoji concept vectors at the same layer
    emoji_cv = concept_vectors[:, 0, sent_target_layer, :]  # (N, d_model)

    print(f"Using layer {sent_target_layer} (index {sent_layer_idx} in sentiment target_layers)")
    print(f"Sentiment CAD vector norm (L{sent_target_layer}): {sent_cad.norm():.4f}")
    print(f"Sentiment prompt vector norm (L{sent_target_layer}): {sent_prompt.norm():.4f}")
    print(f"Emoji CV shape at L{sent_target_layer}: {emoji_cv.shape}")

    # Cosine similarity
    cos_cad = F.cosine_similarity(
        emoji_cv.float(), sent_cad.unsqueeze(0).float(), dim=1
    )  # (N,)
    cos_prompt = F.cosine_similarity(
        emoji_cv.float(), sent_prompt.unsqueeze(0).float(), dim=1
    )  # (N,)

    assert cos_cad.shape == (n_emoji,)
    assert cos_prompt.shape == (n_emoji,)
    print(f"Cosine similarity (CAD) range: [{cos_cad.min():.4f}, {cos_cad.max():.4f}]")
    print(f"Cosine similarity (prompt) range: [{cos_prompt.min():.4f}, {cos_prompt.max():.4f}]")
    print()

    # Step 7: Save outputs
    print("=== Step 7: Saving outputs ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations
    torch.save(individual_activations, output_dir / "all_activations.pt")
    print(f"Saved all_activations.pt: {individual_activations.shape}")

    # Save concept vectors
    torch.save(concept_vectors, output_dir / "all_concept_vectors.pt")
    print(f"Saved all_concept_vectors.pt: {concept_vectors.shape}")

    # Save emoji list
    emoji_list_data = [{"emoji": e, "name": n} for e, n in emoji_data]
    with open(output_dir / "emoji_list.json", "w") as f:
        json.dump(emoji_list_data, f, ensure_ascii=False, indent=2)
    print(f"Saved emoji_list.json: {n_emoji} emoji")

    # Save cosine similarity CSV
    # Build sorted table by CAD cosine similarity (descending)
    rows = []
    for i in range(n_emoji):
        rows.append({
            "emoji": emoji_chars[i],
            "name": emoji_names[i],
            "cos_sim_cad": cos_cad[i].item(),
            "cos_sim_prompt": cos_prompt[i].item(),
        })
    rows.sort(key=lambda r: r["cos_sim_cad"], reverse=True)

    csv_path = output_dir / "sentiment_cosine_similarity.csv"
    with open(csv_path, "w") as f:
        f.write("emoji,name,cos_sim_cad,cos_sim_prompt\n")
        for r in rows:
            # Escape commas in names
            name = r["name"].replace(",", ";")
            f.write(f"{r['emoji']},{name},{r['cos_sim_cad']:.6f},{r['cos_sim_prompt']:.6f}\n")
    print(f"Saved sentiment_cosine_similarity.csv")

    # Save metadata
    total_time = time.time() - start_time
    metadata = {
        "n_emoji": n_emoji,
        "n_layers": n_layers,
        "d_model": d_model,
        "positions": [-1],
        "batch_size": args.batch_size,
        "base_model": args.base_model or DEFAULT_BASE_MODEL,
        "prompt_template": "Tell me about {emoji}",
        "activation_shape": list(individual_activations.shape),
        "concept_vector_shape": list(concept_vectors.shape),
        "sentiment_vectors": {
            "cad_dir": args.sentiment_cad_dir,
            "prompt_dir": args.sentiment_prompt_dir,
            "target_layers": meta_cad["target_layers"],
            "comparison_layer": sent_target_layer,
        },
        "cosine_similarity_stats": {
            "cad_mean": cos_cad.mean().item(),
            "cad_std": cos_cad.std().item(),
            "cad_min": cos_cad.min().item(),
            "cad_max": cos_cad.max().item(),
            "prompt_mean": cos_prompt.mean().item(),
            "prompt_std": cos_prompt.std().item(),
            "prompt_min": cos_prompt.min().item(),
            "prompt_max": cos_prompt.max().item(),
        },
        "timing": {
            "extraction_seconds": round(extraction_time, 1),
            "vector_computation_seconds": round(vector_time, 1),
            "total_seconds": round(total_time, 1),
        },
        "inject_no_think": args.inject_no_think,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata.json")
    print()

    # Step 8: Print sorted report
    print("=" * 70)
    print("TOP 30 EMOJI BY CAD SENTIMENT COSINE SIMILARITY (most positive)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Emoji':>6}  {'cos_cad':>9}  {'cos_prompt':>11}  Name")
    print("-" * 70)
    for i, r in enumerate(rows[:30]):
        print(f"{i+1:>4}  {r['emoji']:>6}  {r['cos_sim_cad']:>9.4f}  {r['cos_sim_prompt']:>11.4f}  {r['name']}")

    print()
    print("=" * 70)
    print("BOTTOM 30 EMOJI BY CAD SENTIMENT COSINE SIMILARITY (most negative)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Emoji':>6}  {'cos_cad':>9}  {'cos_prompt':>11}  Name")
    print("-" * 70)
    for i, r in enumerate(rows[-30:]):
        rank = n_emoji - 29 + i
        print(f"{rank:>4}  {r['emoji']:>6}  {r['cos_sim_cad']:>9.4f}  {r['cos_sim_prompt']:>11.4f}  {r['name']}")

    print()
    print("=" * 60)
    print(f"Extraction complete in {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
