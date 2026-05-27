"""Core activation extraction logic for concept vector computation."""

import torch
from typing import List, Optional
from tqdm import tqdm

from .hook_utils import add_hooks
from .model_utils import format_trajectory, get_model_block_modules


def get_activations_pre_hook(
    layer: int,
    mean_cache: torch.Tensor,
    sample_cache: Optional[torch.Tensor],
    sample_offset: int,
    n_samples: int,
    positions: List[int],
):
    """
    Create a forward pre-hook that extracts activations at specified positions.

    This hook extracts activations entering each transformer layer and:
    1. Accumulates a running mean across all samples (into mean_cache)
    2. Optionally stores individual sample activations (into sample_cache)

    Args:
        layer: Layer index to accumulate into cache
        mean_cache: Pre-allocated tensor of shape (n_positions, n_layers, d_model)
        sample_cache: Optional tensor of shape (n_samples, n_positions, n_layers, d_model)
                      for storing individual activations. If None, only mean is computed.
        sample_offset: Current sample offset for storing into sample_cache
        n_samples: Total number of samples (for mean computation)
        positions: Token positions to extract (e.g., [-1] for last token)

    Returns:
        Hook function to register with module.register_forward_pre_hook()
    """
    def hook_fn(module, input):
        activation = input[0].clone()

        # FAIL FAST: Validate tensor shape
        assert activation.ndim == 3, f"Expected 3D activation, got {activation.ndim}D"
        batch_size, seq_len, d_model = activation.shape

        assert d_model == mean_cache.shape[-1], (
            f"Hidden size mismatch: activation {d_model} != cache {mean_cache.shape[-1]}"
        )

        # Check sequence is long enough for requested positions
        min_required_len = abs(min(positions))
        assert seq_len >= min_required_len, (
            f"Sequence too short ({seq_len}) for positions {positions} "
            f"(need at least {min_required_len} tokens)"
        )

        # Extract activations at specified positions
        # Shape: (batch_size, n_positions, d_model)
        extracted = activation[:, positions, :]

        # Accumulate mean (convert to mean_cache dtype/device)
        mean_cache[:, layer] += (1.0 / n_samples) * extracted.to(mean_cache).sum(dim=0)

        # Store individual activations if sample_cache is provided
        if sample_cache is not None:
            # sample_cache shape: (n_samples, n_positions, n_layers, d_model)
            sample_cache[sample_offset:sample_offset + batch_size, :, layer, :] = extracted.to(sample_cache)

    return hook_fn


def get_per_sample_position_pre_hook(
    layer: int,
    mean_cache: torch.Tensor,
    sample_cache: Optional[torch.Tensor],
    sample_offset: int,
    n_samples: int,
    padded_position_indices: torch.LongTensor,
):
    """Forward pre-hook capturing activations at a DIFFERENT token position per sample.

    Use this when each sample's pre-generation / channel-onset position lives at a
    different token offset within a left-padded batch (e.g., on-policy rollouts
    where prompt+generation lengths vary per turn).

    Args:
        layer: Layer index to accumulate into mean_cache.
        mean_cache: Pre-allocated tensor of shape (n_layers, d_model) on device.
        sample_cache: Optional tensor of shape (n_samples, n_layers, d_model) on CPU.
        sample_offset: Global sample index of batch[0] (for sample_cache write).
        n_samples: Total number of samples (denominator for the running mean).
        padded_position_indices: (B,) LongTensor — per-sample position in the
            padded batch (i.e. accounting for left-pad). Caller is responsible
            for adding pad-len when sequences are left-padded to a common length.
    """
    def hook_fn(module, input):
        activation = input[0].clone()
        assert activation.ndim == 3, f"Expected 3D activation, got {activation.ndim}D"
        B, S, D = activation.shape
        assert D == mean_cache.shape[-1], (
            f"Hidden size mismatch: activation {D} != cache {mean_cache.shape[-1]}"
        )
        idx = padded_position_indices.to(activation.device)
        assert idx.shape == (B,), f"position indices shape {idx.shape} != (B,) = ({B},)"
        assert (idx >= 0).all() and (idx < S).all(), (
            f"position indices out of range [0, {S}): "
            f"min={idx.min().item()}, max={idx.max().item()}"
        )

        batch_idx = torch.arange(B, device=activation.device)
        extracted = activation[batch_idx, idx, :]  # (B, D)

        mean_cache[layer] += (1.0 / n_samples) * extracted.to(mean_cache).sum(dim=0)

        if sample_cache is not None:
            extracted_cpu = extracted.to(sample_cache)
            for b in range(B):
                sample_cache[sample_offset + b, layer, :] = extracted_cpu[b]

    return hook_fn


def get_activations(
    model,
    tokenizer,
    trajectories: List,
    batch_size: int = 32,
    positions: List[int] = None,
    store_individual: bool = False,
    format_kwargs: dict = None,
) -> tuple:
    """
    Extract activations across all layers at specified positions.

    Args:
        model: Merged LoRA model (must have model.model.layers)
        tokenizer: Tokenizer with left-padding configured
        trajectories: List of chat-format trajectories (list of message dicts)
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1] for last token)
        store_individual: If True, also store per-sample activations
        format_kwargs: Optional dict of extra kwargs passed to format_trajectory()
                       (e.g., {"keep_final_im_end": True})

    Returns:
        tuple: (mean_activations, individual_activations)
            - mean_activations: Tensor of shape (n_positions, n_layers, d_model) dtype float64
            - individual_activations: Tensor of shape (n_samples, n_positions, n_layers, d_model)
                                      dtype float32, or None if store_individual=False

    Raises:
        AssertionError: If inputs are invalid or extraction fails
    """
    if positions is None:
        positions = [-1]

    # FAIL FAST: Validate inputs
    assert len(trajectories) > 0, "No trajectories provided"
    assert batch_size > 0, f"Invalid batch_size: {batch_size}"

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len(trajectories)

    print(f"Extracting activations for {n_samples} samples...")
    print(f"  Positions: {positions}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {d_model}")
    print(f"  Store individual: {store_individual}")

    # Get model block modules for hook registration
    block_modules = get_model_block_modules(model)

    # Pre-allocate accumulator in float64 for numerical stability
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model),
        dtype=torch.float64,
        device=model.device,
    )

    # Optionally pre-allocate storage for individual activations (on CPU to avoid OOM)
    if store_individual:
        individual_activations = torch.zeros(
            (n_samples, n_positions, n_layers, d_model),
            dtype=torch.float32,
            device='cpu',
        )
        print(f"  Individual tensor size: {individual_activations.numel() * 4 / 1e9:.2f} GB")
    else:
        individual_activations = None

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations"):
            batch_trajs = trajectories[i:i + batch_size]

            # Create hooks for this batch (need to update sample_offset per batch)
            fwd_pre_hooks = [
                (block_modules[layer], get_activations_pre_hook(
                    layer=layer,
                    mean_cache=mean_activations,
                    sample_cache=individual_activations,
                    sample_offset=i,
                    n_samples=n_samples,
                    positions=positions,
                ))
                for layer in range(n_layers)
            ]

            # Format trajectories for tokenization
            fmt_kw = format_kwargs or {}
            formatted = [format_trajectory(traj, **fmt_kw) for traj in batch_trajs]

            # Tokenize with left-padding
            inputs = tokenizer(
                formatted,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            # FAIL FAST: Validate tokenization
            assert inputs.input_ids.shape[0] == len(batch_trajs), (
                f"Batch size mismatch after tokenization: "
                f"{inputs.input_ids.shape[0]} != {len(batch_trajs)}"
            )

            # Forward pass with hooks to extract activations
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    # FAIL FAST: Validate output
    expected_shape = (n_positions, n_layers, d_model)
    assert mean_activations.shape == expected_shape, (
        f"Output shape {mean_activations.shape} != expected {expected_shape}"
    )
    assert not mean_activations.isnan().any(), "NaN values in mean activations"
    assert not mean_activations.isinf().any(), "Inf values in mean activations"

    if individual_activations is not None:
        assert not individual_activations.isnan().any(), "NaN values in individual activations"

    return mean_activations, individual_activations


def get_mean_activations(
    model,
    tokenizer,
    trajectories: List,
    batch_size: int = 32,
    positions: List[int] = None,
) -> torch.Tensor:
    """
    Extract mean activations across all layers at specified positions.

    This is a convenience wrapper around get_activations() that only returns the mean.

    Args:
        model: Merged LoRA model (must have model.model.layers)
        tokenizer: Tokenizer with left-padding configured
        trajectories: List of chat-format trajectories (list of message dicts)
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1] for last token)

    Returns:
        Tensor of shape (n_positions, n_layers, d_model) with dtype float64

    Raises:
        AssertionError: If inputs are invalid or extraction fails
    """
    mean_activations, _ = get_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories=trajectories,
        batch_size=batch_size,
        positions=positions,
        store_individual=False,
    )
    return mean_activations


def get_all_tile_activations(
    model,
    tokenizer,
    trajectories_by_tile: dict,
    batch_size: int = 32,
    positions: List[int] = None,
    store_individual: bool = False,
    format_kwargs: dict = None,
) -> dict:
    """
    Extract mean activations for each tile type in a SINGLE forward pass.

    This is more efficient than calling get_activations separately for each tile type,
    as we only need to run forward passes once on all trajectories.

    Args:
        model: Merged LoRA model
        tokenizer: Tokenizer with left-padding configured
        trajectories_by_tile: Dict mapping tile type to list of trajectories
                              e.g., {"LAVA": [...], "GOAL": [...], "PATH": [...]}
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1])
        store_individual: If True, also return per-sample activations

    Returns:
        dict with keys for each tile type, each containing:
            - "mean": Mean activations tensor of shape (n_positions, n_layers, d_model)
            - "activations": Per-sample activations (or None if store_individual=False)
            - "count": Number of samples for this tile type
    """
    if positions is None:
        positions = [-1]

    # FAIL FAST: Validate inputs
    assert len(trajectories_by_tile) > 0, "No trajectories provided"
    for tile_type, trajs in trajectories_by_tile.items():
        assert len(trajs) > 0, f"No trajectories for tile type {tile_type}"

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Build flat list of all trajectories with their tile labels
    all_trajectories = []
    tile_labels = []  # Track which tile type each trajectory belongs to
    tile_offsets = {}  # Track start index for each tile type

    current_offset = 0
    for tile_type, trajs in trajectories_by_tile.items():
        tile_offsets[tile_type] = current_offset
        all_trajectories.extend(trajs)
        tile_labels.extend([tile_type] * len(trajs))
        current_offset += len(trajs)

    n_samples = len(all_trajectories)
    tile_counts = {tile: len(trajs) for tile, trajs in trajectories_by_tile.items()}

    # Validate tile_offsets + tile_counts cover the full range
    assert len(tile_labels) == n_samples, (
        f"tile_labels length {len(tile_labels)} != n_samples {n_samples}"
    )
    for tile_type in trajectories_by_tile:
        offset = tile_offsets[tile_type]
        count = tile_counts[tile_type]
        assert all(tile_labels[offset + j] == tile_type for j in range(count)), (
            f"tile_labels mismatch for {tile_type} at offset {offset}"
        )

    print(f"\n=== Unified activation extraction ===")
    print(f"Total trajectories: {n_samples}")
    for tile_type, count in tile_counts.items():
        print(f"  {tile_type}: {count} (offset={tile_offsets[tile_type]})")
    print(f"Store individual activations: {store_individual}")

    # Get model block modules for hook registration
    block_modules = get_model_block_modules(model)

    # Pre-allocate per-tile accumulators in float64 for numerical stability
    mean_accumulators = {
        tile_type: torch.zeros(
            (n_positions, n_layers, d_model),
            dtype=torch.float64,
            device=model.device,
        )
        for tile_type in trajectories_by_tile.keys()
    }

    # Optionally pre-allocate storage for individual activations (on CPU)
    if store_individual:
        individual_activations = {
            tile_type: torch.zeros(
                (tile_counts[tile_type], n_positions, n_layers, d_model),
                dtype=torch.float32,
                device='cpu',
            )
            for tile_type in trajectories_by_tile.keys()
        }
        total_size = sum(t.numel() * 4 for t in individual_activations.values())
        print(f"  Total individual tensor size: {total_size / 1e9:.2f} GB")
    else:
        individual_activations = {tile_type: None for tile_type in trajectories_by_tile.keys()}

    # Custom hook that routes activations to the correct tile accumulator.
    # Uses tile_offsets to compute per-tile sample indices from global position,
    # avoiding mutable counters that cause race conditions between layer hooks.
    def make_unified_hook(layer: int, batch_tile_labels: List[str], batch_start_idx: int):
        def hook_fn(module, input):
            activation = input[0].clone()
            batch_size_actual, seq_len, d_model_actual = activation.shape

            assert batch_size_actual == len(batch_tile_labels), (
                f"Hook batch size {batch_size_actual} != expected {len(batch_tile_labels)}"
            )

            # Extract activations at specified positions
            extracted = activation[:, positions, :]  # (batch, n_positions, d_model)

            # Route each sample to its tile type accumulator
            for i, tile_type in enumerate(batch_tile_labels):
                sample_activation = extracted[i:i+1]  # Keep batch dim: (1, n_positions, d_model)

                # Accumulate mean
                mean_accumulators[tile_type][:, layer] += (
                    (1.0 / tile_counts[tile_type]) * sample_activation.to(mean_accumulators[tile_type]).squeeze(0)
                )

                # Store individual if requested
                if store_individual:
                    # Compute per-tile sample index from global position
                    global_idx = batch_start_idx + i
                    sample_idx = global_idx - tile_offsets[tile_type]
                    assert 0 <= sample_idx < tile_counts[tile_type], (
                        f"sample_idx {sample_idx} out of range [0, {tile_counts[tile_type]}) "
                        f"for tile {tile_type} (global_idx={global_idx}, offset={tile_offsets[tile_type]})"
                    )
                    individual_activations[tile_type][sample_idx, :, layer, :] = (
                        sample_activation.squeeze(0).to(individual_activations[tile_type])
                    )

        return hook_fn

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations (unified)"):
            batch_trajs = all_trajectories[i:i + batch_size]
            batch_labels = tile_labels[i:i + batch_size]

            assert len(batch_trajs) == len(batch_labels), (
                f"Batch size mismatch: {len(batch_trajs)} trajs vs {len(batch_labels)} labels at offset {i}"
            )

            # Create hooks for this batch
            fwd_pre_hooks = [
                (block_modules[layer], make_unified_hook(layer, batch_labels, i))
                for layer in range(n_layers)
            ]

            # Format trajectories for tokenization
            tile_fmt_kw = format_kwargs or {}
            formatted = [format_trajectory(traj, **tile_fmt_kw) for traj in batch_trajs]

            # Tokenize with left-padding
            inputs = tokenizer(
                formatted,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            # Forward pass with hooks
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    # Validate outputs
    for tile_type, mean_act in mean_accumulators.items():
        assert not mean_act.isnan().any(), f"NaN values in mean activations for {tile_type}"
        assert not mean_act.isinf().any(), f"Inf values in mean activations for {tile_type}"

    # Build result dict
    results = {}
    for tile_type in trajectories_by_tile.keys():
        results[tile_type] = {
            "mean": mean_accumulators[tile_type],
            "activations": individual_activations[tile_type],
            "count": tile_counts[tile_type],
        }

    # Report statistics
    print(f"\n=== Extraction complete ===")
    for tile_type, data in results.items():
        norm = data["mean"].norm(dim=-1).mean()
        print(f"  {tile_type}: count={data['count']}, mean_norm={norm:.4f}")

    return results


def get_mean_diff(
    model,
    tokenizer,
    lava_trajectories: List,
    other_trajectories: List,
    batch_size: int = 32,
    positions: List[int] = None,
    store_individual: bool = False,
    format_kwargs: dict = None,
) -> tuple:
    """
    Compute the mean activation difference between LAVA and OTHER trajectories.

    This is the core concept vector computation:
    concept_vector = mean_activations_LAVA - mean_activations_OTHER

    Args:
        model: Merged LoRA model
        tokenizer: Tokenizer with left-padding configured
        lava_trajectories: Trajectories ending on LAVA tiles
        other_trajectories: Trajectories ending on PATH or GOAL tiles
        batch_size: Batch size for forward passes
        positions: Token positions to extract (default: [-1])
        store_individual: If True, also return per-sample activations for both classes

    Returns:
        tuple: (mean_diff, mean_lava, mean_other, activations_lava, activations_other)
            - mean_diff: Concept vector tensor of shape (n_positions, n_layers, d_model)
            - mean_lava: Mean activations for LAVA class
            - mean_other: Mean activations for OTHER class
            - activations_lava: Per-sample activations for LAVA class (or None)
            - activations_other: Per-sample activations for OTHER class (or None)

    Raises:
        AssertionError: If inputs are invalid or computation fails
    """
    if positions is None:
        positions = [-1]

    # FAIL FAST: Validate inputs
    assert len(lava_trajectories) > 0, "No LAVA trajectories provided"
    assert len(other_trajectories) > 0, "No OTHER trajectories provided"

    print(f"\n=== Computing concept vector ===")
    print(f"LAVA trajectories: {len(lava_trajectories)}")
    print(f"OTHER trajectories: {len(other_trajectories)}")
    print(f"Store individual activations: {store_individual}")

    # Extract activations for each class
    print("\n--- Extracting LAVA activations ---")
    mean_lava, activations_lava = get_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories=lava_trajectories,
        batch_size=batch_size,
        positions=positions,
        store_individual=store_individual,
        format_kwargs=format_kwargs,
    )

    # Clear CUDA cache between extractions to avoid OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Extracting OTHER activations ---")
    mean_other, activations_other = get_activations(
        model=model,
        tokenizer=tokenizer,
        trajectories=other_trajectories,
        batch_size=batch_size,
        positions=positions,
        store_individual=store_individual,
        format_kwargs=format_kwargs,
    )

    # FAIL FAST: Validate shapes match
    assert mean_lava.shape == mean_other.shape, (
        f"Shape mismatch: LAVA {mean_lava.shape} != OTHER {mean_other.shape}"
    )

    # Compute concept vector
    mean_diff = mean_lava - mean_other

    # FAIL FAST: Validate output
    assert not mean_diff.isnan().any(), "NaN values in mean_diff"
    assert not mean_diff.isinf().any(), "Inf values in mean_diff"

    # Report statistics
    diff_norms = mean_diff.norm(dim=-1)  # (n_positions, n_layers)
    print(f"\n=== Concept vector statistics ===")
    print(f"Shape: {mean_diff.shape}")
    print(f"Norm range per layer: [{diff_norms.min():.4f}, {diff_norms.max():.4f}]")
    print(f"Mean norm: {diff_norms.mean():.4f}")

    return mean_diff, mean_lava, mean_other, activations_lava, activations_other
