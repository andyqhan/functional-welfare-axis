"""
Cosine-similarity comparison: VAA vs the existing concept-vector library on
Qwen3-4B-Instruct-2507 (and LoRA/FFT fine-tunes that share the same residual
stream).

Inputs read:
  vaa/  artifact:   artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa/mean_diff.pt
  comparison:       artifacts/concept_vectors/baseline_office_emoji/baseline/{lava,goal,path}/mean_diff.pt
                    artifacts/concept_vectors/<sweep_run>/off_policy/{lava,goal,path}/mean_diff.pt  (10 runs)
                    artifacts/sentiment_vectors/mean_diff.pt   (CAD; layers 20-23 only)
                    artifacts/sentiment_vectors_prompt/mean_diff.pt   (prompt; layers 20-23 only)

Outputs:
  vaa/analysis/cosine_vs_concept_vectors.csv
  vaa/analysis/cosine_vs_concept_vectors.png   (bar plot at the picked layer)
  vaa/analysis/cosine_per_layer.csv            (full layer-by-layer matrix where comparable)

Pure CPU. Run on a login node.
"""

import argparse
import csv
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts"

VAA_DIR = ART / "concept_vectors" / "vaa_qwen3_4b_instruct" / "baseline" / "vaa"

# Sweep checkpoints -> base model (from CLAUDE.md). We only cosine-compare
# vectors that live in the SAME residual stream as VAA (Qwen3-4B-Instruct-2507).
SWEEP_RUNS_4B_INSTRUCT = [
    "pytorch-grpo-4b-2602551_step95",        # 4B DrGRPO
    "pytorch-grpo-4b-2694182_step50",        # 4B DrGRPO office-swap
    "sft-test-5678169_step9375",             # SFT
    "token-reinforce-full-4081097_step115",  # Token REINFORCE
    "sft-instruct-fft_step3125",             # 4B Instruct FFT SFT
    "pytorch-grpo-4b-fft_step95",            # 4B Instruct FFT RL
]
# 4B-Base, 8B-Cardinal, and Tinker-v5 use different residual streams -> excluded.

CONCEPTS = ["lava", "goal", "path"]

# Sentiment vectors (only layers 20-23) — built on Qwen3-4B-Instruct
SENTIMENT_VECTOR_DIRS = [
    ("sentiment_cad", ART / "sentiment_vectors", [20, 21, 22, 23]),
    ("sentiment_prompt", ART / "sentiment_vectors_prompt", [20, 21, 22, 23]),
]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    na = a.norm()
    nb = b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def load_diff(path: Path):
    """Returns the (n_layers, d_model) view of a mean_diff.pt (drops position dim)."""
    if not path.exists():
        return None
    t = torch.load(path, weights_only=True, map_location="cpu")
    if t.ndim == 3 and t.shape[0] == 1:
        return t[0]
    if t.ndim == 2:
        return t
    raise ValueError(f"unexpected mean_diff shape at {path}: {tuple(t.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "vaa" / "analysis"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vaa = load_diff(VAA_DIR / "mean_diff.pt")
    assert vaa is not None, f"VAA not extracted yet: {VAA_DIR}"
    n_layers, d_model = vaa.shape
    print(f"VAA loaded: ({n_layers}, {d_model})")

    meta = json.load(open(VAA_DIR / "metadata.json"))
    auto_layer = int(meta["auto_selected_layer"])
    print(f"VAA auto-selected layer: {auto_layer}")

    rows = []  # for the picked-layer CSV

    # --- Pre-RL baseline (matched residual stream)
    for c in CONCEPTS:
        diff = load_diff(ART / "concept_vectors" / "baseline_office_emoji" / "baseline" / c / "mean_diff.pt")
        if diff is None:
            continue
        cos = cosine(vaa[auto_layer], diff[auto_layer])
        rows.append(("baseline_office_emoji", "baseline", c, auto_layer, cos))

    # --- Trained 4B-Instruct sweep runs
    for run in SWEEP_RUNS_4B_INSTRUCT:
        for c in CONCEPTS:
            diff = load_diff(ART / "concept_vectors" / run / "off_policy" / c / "mean_diff.pt")
            if diff is None:
                continue
            cos = cosine(vaa[auto_layer], diff[auto_layer])
            rows.append((run, "off_policy", c, auto_layer, cos))

    # --- Sentiment vectors (only at layers 20-23)
    for label, sdir, layers in SENTIMENT_VECTOR_DIRS:
        diff = load_diff(sdir / "mean_diff.pt")
        if diff is None:
            print(f"  ! {label} not found at {sdir}")
            continue
        # Sentiment artifacts store only 4 layers; map our auto_layer to the closest
        if auto_layer in layers:
            sent_idx = layers.index(auto_layer)
            cos = cosine(vaa[auto_layer], diff[sent_idx])
            rows.append((label, "sentiment", label, auto_layer, cos))
        else:
            # Fall back: report at each available layer
            for sent_idx, lyr in enumerate(layers):
                cos = cosine(vaa[lyr], diff[sent_idx])
                rows.append((label, "sentiment", label, lyr, cos))

    # Write picked-layer CSV
    csv_path = out_dir / "cosine_vs_concept_vectors.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "group", "concept", "layer", "cosine"])
        for r in sorted(rows, key=lambda r: -abs(r[4]) if r[4] == r[4] else 1.0):
            w.writerow(r)
    print(f"\nWrote {csv_path}  ({len(rows)} rows)")

    # Print top hits
    print("\nTop |cosine| (VAA vs other concept vectors):")
    for r in sorted(rows, key=lambda r: -abs(r[4]) if r[4] == r[4] else 1.0)[:25]:
        print(f"  {r[4]:+.4f}  layer={r[3]}  {r[0]}/{r[1]}/{r[2]}")

    # Per-layer matrix where comparable (full residual-stream vectors at all 36 layers)
    per_layer_path = out_dir / "cosine_per_layer.csv"
    with open(per_layer_path, "w") as f:
        w = csv.writer(f)
        header = ["artifact_concept"] + [f"L{lyr}" for lyr in range(n_layers)]
        w.writerow(header)
        # Pre-RL baseline
        for c in CONCEPTS:
            diff = load_diff(ART / "concept_vectors" / "baseline_office_emoji" / "baseline" / c / "mean_diff.pt")
            if diff is None:
                continue
            row = [f"baseline_office_emoji/{c}"] + [f"{cosine(vaa[l], diff[l]):.4f}" for l in range(n_layers)]
            w.writerow(row)
        # Trained
        for run in SWEEP_RUNS_4B_INSTRUCT:
            for c in CONCEPTS:
                diff = load_diff(ART / "concept_vectors" / run / "off_policy" / c / "mean_diff.pt")
                if diff is None:
                    continue
                row = [f"{run}/{c}"] + [f"{cosine(vaa[l], diff[l]):.4f}" for l in range(n_layers)]
                w.writerow(row)
    print(f"Wrote {per_layer_path}")

    # Quick bar plot
    try:
        import matplotlib.pyplot as plt
        picked_rows = [r for r in rows if r[3] == auto_layer]
        picked_rows.sort(key=lambda r: r[4])
        labels = [f"{r[0][:30]}/{r[2]}" for r in picked_rows]
        vals = [r[4] for r in picked_rows]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.25 * len(labels))))
        ax.barh(labels, vals)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"cosine(VAA, *) at layer {auto_layer}")
        ax.set_title("VAA vs maze concept vectors (Qwen3-4B-Instruct residual stream)")
        plt.tight_layout()
        png_path = out_dir / "cosine_vs_concept_vectors.png"
        plt.savefig(png_path, dpi=140)
        print(f"Wrote {png_path}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
