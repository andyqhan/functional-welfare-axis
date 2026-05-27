"""
Extract emotion concept vectors from generated stories.

Three phases:
  A) Extract per-emotion mean activations (averaged over token positions 50+)
  B) Subtract global mean across all emotions
  C) PCA denoising using neutral dialogue activations

Usage:
    uv run python -m emotions.extract_emotion_vectors \
        --model Qwen/Qwen3-4B-Instruct-2507 --phase all

    uv run python -m emotions.extract_emotion_vectors \
        --model Qwen/Qwen3-4B-Instruct-2507 --phase extract

    uv run python -m emotions.extract_emotion_vectors \
        --model Qwen/Qwen3-4B-Instruct-2507 --phase denoise
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from emotions.emotion_utils import get_artifact_dir, load_emotions
from src.concept_vector.hook_utils import add_hooks
from src.concept_vector.model_utils import get_model_block_modules, load_model_auto


# ─── Token-Averaging Hook ───────────────────────────────────────────────────


def compute_averaging_weights(
    attention_mask: torch.Tensor,
    content_start_offset: int = 50,
) -> torch.Tensor:
    """Pre-compute per-sample token averaging weights.

    For each sample in the batch:
    1. Find content start (first non-pad token via attention_mask)
    2. Averaging range: [content_start + offset, seq_end]
    3. Weight = 1/n_valid_tokens for tokens in range, 0 otherwise

    If a story has fewer than `offset` content tokens, falls back to
    averaging all content tokens (offset=0).

    Args:
        attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
        content_start_offset: Skip first N content tokens (default: 50)

    Returns:
        weights: (batch_size, seq_len), float32. Each row sums to 1.0
                 (or 0.0 if the sample has no content, which shouldn't happen).
    """
    batch_size, seq_len = attention_mask.shape
    weights = torch.zeros(batch_size, seq_len, dtype=torch.float32,
                          device=attention_mask.device)

    for i in range(batch_size):
        nonpad = attention_mask[i].nonzero(as_tuple=True)[0]
        assert len(nonpad) > 0, f"Sample {i} has no content tokens"

        content_start = nonpad[0].item()
        content_len = len(nonpad)

        # Apply offset, fall back to all content if too short
        if content_len > content_start_offset:
            avg_start = content_start + content_start_offset
        else:
            avg_start = content_start

        # Only weight positions that are actual content (attention_mask=1).
        # With left-padding all positions from avg_start onward are content,
        # but using the mask makes this robust to any padding scheme.
        range_mask = attention_mask[i, avg_start:].float()
        n_tokens = int(range_mask.sum().item())
        assert n_tokens > 0, (
            f"Sample {i}: no tokens to average (seq_len={seq_len}, "
            f"avg_start={avg_start}, content_len={content_len})"
        )
        weights[i, avg_start:] = range_mask / n_tokens

    return weights


def make_token_averaging_hook(
    layer_idx: int,
    mean_cache: torch.Tensor,
    n_samples: int,
    weights: torch.Tensor,
):
    """Create a forward pre-hook that averages activations using pre-computed weights.

    The hook body is a single einsum + accumulation — fully vectorized.

    Args:
        layer_idx: Index into mean_cache's layer dimension
        mean_cache: (n_layers, d_model) float64, running mean accumulator
        n_samples: Total samples for running mean normalization
        weights: (batch_size, seq_len) float32, pre-computed averaging weights
    """
    def hook_fn(module, input):
        activation = input[0]  # (batch_size, seq_len, d_model)
        # Weighted average per sample: (batch_size, d_model)
        weighted = torch.einsum(
            "bs,bsd->bd", weights.to(activation.device, activation.dtype), activation
        )
        # Accumulate into running mean
        mean_cache[layer_idx] += (1.0 / n_samples) * weighted.to(mean_cache).sum(dim=0)

    return hook_fn


def make_token_averaging_hook_individual(
    layer_idx: int,
    individual_cache: torch.Tensor,
    sample_offset: int,
    weights: torch.Tensor,
):
    """Create a hook that stores per-sample token-averaged activations.

    Used for neutral dialogues where we need individual activations for PCA.

    Args:
        layer_idx: Index into individual_cache's layer dimension
        individual_cache: (n_samples, n_layers, d_model) float32
        sample_offset: Global sample offset for this batch
        weights: (batch_size, seq_len) float32
    """
    def hook_fn(module, input):
        activation = input[0]  # (batch_size, seq_len, d_model)
        weighted = torch.einsum(
            "bs,bsd->bd", weights.to(activation.device, activation.dtype), activation
        )
        batch_size = weighted.shape[0]
        individual_cache[sample_offset:sample_offset + batch_size, layer_idx] = (
            weighted.to(individual_cache)
        )

    return hook_fn


# ─── Phase A: Extract Per-Emotion Means ─────────────────────────────────────


def load_stories(jsonl_path: Path) -> list[str]:
    """Load story texts from a JSONL file."""
    stories = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            stories.append(entry["story"])
    return stories


def extract_emotion_means(
    model, tokenizer, emotions: list[str], artifact_dir: Path,
    batch_size: int = 64, content_start_offset: int = 50,
) -> torch.Tensor:
    """Extract mean activation per emotion, averaged over token positions 50+.

    Returns:
        emotion_means: (n_emotions, n_layers, d_model) float64
    """
    block_modules = get_model_block_modules(model)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_emotions = len(emotions)

    emotion_means = torch.zeros(
        n_emotions, n_layers, d_model,
        dtype=torch.float64, device="cpu",
    )

    stories_dir = artifact_dir / "stories"

    for emo_idx, emotion in enumerate(emotions):
        safe_name = emotion.replace(" ", "_")
        jsonl_path = stories_dir / f"{safe_name}.jsonl"
        assert jsonl_path.exists(), f"Stories file not found: {jsonl_path}"

        stories = load_stories(jsonl_path)
        n_stories = len(stories)
        assert n_stories > 0, f"No stories for emotion '{emotion}'"
        print(f"  [{emo_idx + 1}/{n_emotions}] {emotion}: {n_stories} stories", flush=True)

        # Per-emotion accumulator on GPU
        mean_cache = torch.zeros(n_layers, d_model, dtype=torch.float64, device=model.device)

        with torch.no_grad():
            for i in range(0, n_stories, batch_size):
                batch_stories = stories[i:i + batch_size]

                # Tokenize raw text (no chat template)
                inputs = tokenizer(
                    batch_stories, padding=True, truncation=False,
                    return_tensors="pt", add_special_tokens=True,
                )
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)

                # Pre-compute averaging weights for this batch
                weights = compute_averaging_weights(attention_mask, content_start_offset)

                # Register hooks on all layers
                fwd_pre_hooks = [
                    (
                        block_modules[layer],
                        make_token_averaging_hook(layer, mean_cache, n_stories, weights),
                    )
                    for layer in range(n_layers)
                ]

                with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                    model(input_ids=input_ids, attention_mask=attention_mask)

        # Store on CPU
        emotion_means[emo_idx] = mean_cache.cpu()

        # Validate
        assert not emotion_means[emo_idx].isnan().any(), f"NaN in means for '{emotion}'"
        assert not emotion_means[emo_idx].isinf().any(), f"Inf in means for '{emotion}'"

        # Clear GPU cache periodically
        torch.cuda.empty_cache()

    return emotion_means


# ─── Phase C: PCA Denoising ─────────────────────────────────────────────────


def load_neutral_dialogues(jsonl_path: Path) -> list[str]:
    """Load neutral dialogue texts from JSONL."""
    dialogues = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            dialogues.append(entry["dialogue"])
    return dialogues


def extract_neutral_activations(
    model, tokenizer, dialogues: list[str],
    batch_size: int = 64, content_start_offset: int = 50,
) -> torch.Tensor:
    """Extract individual token-averaged activations for each neutral dialogue.

    Returns:
        activations: (n_dialogues, n_layers, d_model) float32, on CPU
    """
    block_modules = get_model_block_modules(model)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_dialogues = len(dialogues)

    # Store individual activations on CPU
    individual = torch.zeros(
        n_dialogues, n_layers, d_model,
        dtype=torch.float32, device="cpu",
    )

    print(f"  Extracting activations for {n_dialogues} neutral dialogues...")
    with torch.no_grad():
        for i in tqdm(range(0, n_dialogues, batch_size), desc="Neutral extraction"):
            batch = dialogues[i:i + batch_size]

            # Tokenize raw text
            inputs = tokenizer(
                batch, padding=True, truncation=False,
                return_tensors="pt", add_special_tokens=True,
            )
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            weights = compute_averaging_weights(attention_mask, content_start_offset)

            fwd_pre_hooks = [
                (
                    block_modules[layer],
                    make_token_averaging_hook_individual(layer, individual, i, weights),
                )
                for layer in range(n_layers)
            ]

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(input_ids=input_ids, attention_mask=attention_mask)

    assert not individual.isnan().any(), "NaN in neutral activations"
    assert not individual.isinf().any(), "Inf in neutral activations"
    return individual


def pca_denoise(
    emotion_vectors: torch.Tensor,
    neutral_activations: torch.Tensor,
    variance_threshold: float = 0.50,
) -> tuple[torch.Tensor, dict, list]:
    """Project out top PCA components of neutral activations from emotion vectors.

    Per-layer:
    1. Center neutral activations
    2. Compute covariance, eigendecompose
    3. Find top-k components explaining >= variance_threshold of total variance
    4. Project out: v' = v - V_k @ V_k^T @ v

    Args:
        emotion_vectors: (n_emotions, n_layers, d_model) — centered emotion vectors
        neutral_activations: (n_neutral, n_layers, d_model) — individual neutral activations
        variance_threshold: Cumulative variance explained threshold (default: 0.50)

    Returns:
        denoised: (n_emotions, n_layers, d_model) — denoised emotion vectors
        pca_info: dict with per-layer info (n_components, variance_explained)
    """
    n_emotions, n_layers, d_model = emotion_vectors.shape
    n_neutral = neutral_activations.shape[0]

    # Work in float64 for numerical stability
    emotion_f64 = emotion_vectors.to(torch.float64)
    neutral_f64 = neutral_activations.to(torch.float64)

    denoised = torch.zeros_like(emotion_f64)
    pca_info = {"per_layer": []}
    all_components = []

    print(f"  PCA denoising: {n_neutral} neutral samples, threshold={variance_threshold}")
    for layer in tqdm(range(n_layers), desc="PCA denoising per layer"):
        # Center neutral activations at this layer
        neutral_layer = neutral_f64[:, layer, :]  # (n_neutral, d_model)
        neutral_mean = neutral_layer.mean(dim=0)
        centered = neutral_layer - neutral_mean

        # Covariance matrix: (d_model, d_model)
        cov = (centered.T @ centered) / (n_neutral - 1)

        # Eigendecomposition (eigh returns ascending eigenvalues)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Reverse to descending order
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Find k components that explain >= threshold of variance
        total_variance = eigenvalues.sum()
        assert total_variance > 0, f"Layer {layer}: zero total variance"
        cumulative_ratio = eigenvalues.cumsum(0) / total_variance

        # Smallest k such that cumulative_ratio[k-1] >= threshold
        k = (cumulative_ratio >= variance_threshold).nonzero(as_tuple=True)[0]
        assert len(k) > 0, f"Layer {layer}: can't reach {variance_threshold} variance"
        k = k[0].item() + 1  # +1 because we want the count, not the index

        # Top-k eigenvectors: (d_model, k)
        V_k = eigenvectors[:, :k]

        # Project out: v' = v - V_k @ V_k^T @ v
        emotion_layer = emotion_f64[:, layer, :]  # (n_emotions, d_model)
        projection = emotion_layer @ V_k @ V_k.T  # (n_emotions, d_model)
        denoised[:, layer, :] = emotion_layer - projection

        actual_variance = cumulative_ratio[k - 1].item()
        pca_info["per_layer"].append({
            "layer": layer,
            "n_components": k,
            "variance_explained": round(actual_variance, 4),
            "total_eigenvalues": eigenvalues.shape[0],
        })
        all_components.append(V_k.cpu())

    # Summary statistics
    n_components_list = [info["n_components"] for info in pca_info["per_layer"]]
    pca_info["summary"] = {
        "variance_threshold": variance_threshold,
        "n_neutral_samples": n_neutral,
        "mean_n_components": round(sum(n_components_list) / len(n_components_list), 1),
        "min_n_components": min(n_components_list),
        "max_n_components": max(n_components_list),
    }

    print(f"  PCA components per layer: min={min(n_components_list)}, "
          f"max={max(n_components_list)}, mean={pca_info['summary']['mean_n_components']}")

    return denoised, pca_info, all_components


# ─── Sanity Check ────────────────────────────────────────────────────────────


def sanity_check(emotion_vectors: torch.Tensor, emotions: list[str], target_layer: int):
    """Print top-5 most and least similar emotion pairs at the target layer."""
    vecs = emotion_vectors[:, target_layer, :].to(torch.float32)
    # Normalize
    norms = vecs.norm(dim=-1, keepdim=True)
    vecs_normed = vecs / norms.clamp(min=1e-8)
    # Cosine similarity matrix
    sim = vecs_normed @ vecs_normed.T

    n = len(emotions)
    # Collect upper triangle
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim[i, j].item(), emotions[i], emotions[j]))

    pairs.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Sanity check (layer {target_layer}):")
    print(f"  Top-5 most similar pairs:")
    for score, e1, e2 in pairs[:5]:
        print(f"    {score:.4f}  {e1} / {e2}")
    print(f"  Top-5 least similar pairs:")
    for score, e1, e2 in pairs[-5:]:
        print(f"    {score:.4f}  {e1} / {e2}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract emotion concept vectors")
    parser.add_argument("--model", required=True, help="HF model name or checkpoint path")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional local checkpoint path (for LoRA models)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--content-start-offset", type=int, default=50,
                        help="Skip first N content tokens when averaging (default: 50)")
    parser.add_argument("--phase", choices=["extract", "denoise", "all"], default="all")
    parser.add_argument("--variance-threshold", type=float, default=0.50,
                        help="PCA variance threshold for denoising (default: 0.50)")
    args = parser.parse_args()

    t0 = time.time()
    emotions = load_emotions()
    artifact_dir = get_artifact_dir(args.model)

    print("=" * 70)
    print("Emotion Concept Vector Extraction")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint or '(base model)'}")
    print(f"  Artifact dir: {artifact_dir}")
    print(f"  Phase: {args.phase}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Content start offset: {args.content_start_offset}")
    print(f"  Variance threshold: {args.variance_threshold}")
    print()

    # Save emotions order (index -> emotion mapping)
    emotions_order_path = artifact_dir / "emotions_order.json"
    if not emotions_order_path.exists():
        with open(emotions_order_path, "w") as f:
            json.dump(emotions, f, indent=2)
        print(f"  Saved emotions order to {emotions_order_path}")

    # ── Phase A + B: Extract ──────────────────────────────────────────────

    raw_dir = artifact_dir / "raw_vectors"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("extract", "all"):
        print("\n" + "=" * 70)
        print("Phase A: Extracting per-emotion mean activations")
        print("=" * 70)

        # Load model
        model_path = args.checkpoint or args.model
        print(f"Loading model: {model_path}")
        model, tokenizer = load_model_auto(model_path)
        n_layers = model.config.num_hidden_layers
        d_model = model.config.hidden_size
        print(f"  Layers: {n_layers}, Hidden: {d_model}")

        emotion_means = extract_emotion_means(
            model, tokenizer, emotions, artifact_dir,
            batch_size=args.batch_size,
            content_start_offset=args.content_start_offset,
        )

        # Phase B: Global mean subtraction
        print("\nPhase B: Computing global mean and centered vectors")
        global_mean = emotion_means.mean(dim=0)  # (n_layers, d_model)
        emotion_vectors_raw = emotion_means - global_mean.unsqueeze(0)

        # Save
        torch.save(emotion_means, raw_dir / "emotion_means.pt")
        torch.save(global_mean, raw_dir / "global_mean.pt")
        torch.save(emotion_vectors_raw, raw_dir / "emotion_vectors_raw.pt")

        metadata = {
            "n_emotions": len(emotions),
            "n_layers": n_layers,
            "d_model": d_model,
            "content_start_offset": args.content_start_offset,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "shapes": {
                "emotion_means": list(emotion_means.shape),
                "global_mean": list(global_mean.shape),
                "emotion_vectors_raw": list(emotion_vectors_raw.shape),
            },
            "dtype": str(emotion_means.dtype),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(raw_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {raw_dir}/")
        print(f"  emotion_means: {emotion_means.shape}")
        print(f"  global_mean: {global_mean.shape}")
        print(f"  emotion_vectors_raw: {emotion_vectors_raw.shape}")
        print(f"  Mean norm (raw vectors): {emotion_vectors_raw.norm(dim=-1).mean():.4f}")

        # Sanity check on raw vectors
        target_layer = int(n_layers * 2 / 3)
        sanity_check(emotion_vectors_raw, emotions, target_layer)

        # Free model for denoise phase if running all
        if args.phase == "all":
            # Keep model loaded — we need it for neutral extraction
            pass
        else:
            del model
            torch.cuda.empty_cache()

    # ── Phase C: PCA Denoising ────────────────────────────────────────────

    denoised_dir = artifact_dir / "denoised_vectors"
    denoised_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("denoise", "all"):
        print("\n" + "=" * 70)
        print("Phase C: PCA denoising with neutral dialogues")
        print("=" * 70)

        # Load raw vectors if not already in memory
        if args.phase == "denoise":
            raw_path = raw_dir / "emotion_vectors_raw.pt"
            assert raw_path.exists(), f"Raw vectors not found: {raw_path}. Run --phase extract first."
            emotion_vectors_raw = torch.load(raw_path, weights_only=True)
            print(f"  Loaded raw vectors: {emotion_vectors_raw.shape}")

            # Load model for neutral extraction
            model_path = args.checkpoint or args.model
            print(f"Loading model: {model_path}")
            model, tokenizer = load_model_auto(model_path)

        n_layers = model.config.num_hidden_layers
        d_model = model.config.hidden_size

        # Load neutral dialogues
        neutral_path = artifact_dir / "neutral" / "neutral_dialogues.jsonl"
        assert neutral_path.exists(), f"Neutral dialogues not found: {neutral_path}"
        dialogues = load_neutral_dialogues(neutral_path)
        print(f"  Loaded {len(dialogues)} neutral dialogues")

        # Extract individual activations
        neutral_activations = extract_neutral_activations(
            model, tokenizer, dialogues,
            batch_size=args.batch_size,
            content_start_offset=args.content_start_offset,
        )
        print(f"  Neutral activations: {neutral_activations.shape}")

        # Free model
        del model
        torch.cuda.empty_cache()

        # PCA denoise
        denoised, pca_info, pca_components = pca_denoise(
            emotion_vectors_raw, neutral_activations,
            variance_threshold=args.variance_threshold,
        )

        # Save
        torch.save(denoised, denoised_dir / "emotion_vectors.pt")
        torch.save(pca_components, denoised_dir / "pca_components.pt")

        with open(denoised_dir / "pca_info.json", "w") as f:
            json.dump(pca_info, f, indent=2)

        elapsed = time.time() - t0
        metadata = {
            "n_emotions": len(emotions),
            "n_layers": n_layers,
            "d_model": d_model,
            "content_start_offset": args.content_start_offset,
            "variance_threshold": args.variance_threshold,
            "n_neutral_dialogues": len(dialogues),
            "model": args.model,
            "checkpoint": args.checkpoint,
            "shapes": {
                "emotion_vectors": list(denoised.shape),
            },
            "dtype": str(denoised.dtype),
            "pca_summary": pca_info["summary"],
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(denoised_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  Saved to {denoised_dir}/")
        print(f"  emotion_vectors.pt: {denoised.shape}")
        print(f"  Mean norm (denoised): {denoised.norm(dim=-1).mean():.4f}")

        # Sanity check on denoised vectors
        target_layer = int(n_layers * 2 / 3)
        sanity_check(denoised, emotions, target_layer)

    # Final summary
    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"Extraction complete in {elapsed / 60:.1f}min")
    print(f"  Artifact dir: {artifact_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
