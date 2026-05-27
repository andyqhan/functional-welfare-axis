"""Shared output writer for concept vector extraction methods.

Enforces a consistent output contract: each concept gets a subdirectory
with mean_diff.pt, activations_lava.pt, activations_other.pt, metadata.json.
A concepts.txt file lists all extracted concepts.
"""

import json
import torch
from pathlib import Path
from typing import TypedDict


class ConceptResult(TypedDict):
    mean_diff: torch.Tensor  # (n_pos, n_layers, d_model), float64
    activations_positive: torch.Tensor | None  # (n, n_pos, n_layers, d_model), float32
    activations_negative: torch.Tensor | None  # (n, n_pos, n_layers, d_model), float32
    metadata: dict  # method-specific metadata


def write_extraction_outputs(
    output_dir: Path,
    concepts: dict[str, ConceptResult],
) -> list[str]:
    """Write extraction outputs for all concepts in a standard format.

    Creates <output_dir>/<concept>/ for each concept with:
      - mean_diff.pt: concept vector (n_pos, n_layers, d_model), float64
      - activations_lava.pt: positive class activations (optional)
      - activations_other.pt: negative class activations (optional)
      - metadata.json: method-specific metadata

    Also writes <output_dir>/concepts.txt with one concept name per line.

    Args:
        output_dir: Base output directory
        concepts: Mapping of concept name -> ConceptResult

    Returns:
        List of concept names written
    """
    output_dir = Path(output_dir)
    assert len(concepts) > 0, "No concepts to write"

    # Validate all concepts before writing anything
    d_model = None
    for name, result in concepts.items():
        md = result["mean_diff"]
        assert md.ndim == 3, (
            f"Concept '{name}': mean_diff must be 3D (n_pos, n_layers, d_model), "
            f"got {md.ndim}D with shape {md.shape}"
        )
        assert not md.isnan().any(), f"Concept '{name}': NaN in mean_diff"
        assert not md.isinf().any(), f"Concept '{name}': Inf in mean_diff"

        if d_model is None:
            d_model = md.shape[-1]
        else:
            assert md.shape[-1] == d_model, (
                f"Concept '{name}': d_model mismatch {md.shape[-1]} != {d_model}"
            )

    # Write outputs
    concept_names = []
    for name, result in concepts.items():
        concept_dir = output_dir / name
        concept_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors (move to CPU for storage)
        torch.save(result["mean_diff"].cpu(), concept_dir / "mean_diff.pt")

        if result["activations_positive"] is not None:
            torch.save(
                result["activations_positive"].cpu(),
                concept_dir / "activations_lava.pt",
            )
        if result["activations_negative"] is not None:
            torch.save(
                result["activations_negative"].cpu(),
                concept_dir / "activations_other.pt",
            )

        # Save metadata
        with open(concept_dir / "metadata.json", "w") as f:
            json.dump(result["metadata"], f, indent=2)

        # Report
        md = result["mean_diff"]
        diff_norms = md.norm(dim=-1).squeeze()
        print(f"  {name.upper()} concept vector:")
        print(f"    Shape: {md.shape}")
        print(f"    Layer norm range: [{diff_norms.min():.4f}, {diff_norms.max():.4f}]")
        print(f"    Top 5 layers by norm: {diff_norms.argsort(descending=True)[:5].tolist()}")
        print(f"    Saved to: {concept_dir}")

        concept_names.append(name)

    # Write concepts.txt
    concepts_file = output_dir / "concepts.txt"
    concepts_file.write_text("\n".join(concept_names) + "\n")
    print(f"\nWrote {len(concept_names)} concepts to {concepts_file}")

    return concept_names
