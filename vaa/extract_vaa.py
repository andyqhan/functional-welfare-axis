"""
Extract the Valence-Assent Axis (Lu et al. 2025, arXiv:2510.27328) on
Qwen3-4B-Instruct-2507 so the existing concept-vector eval pipelines can
consume it.

Recipe:
  1. Load the base model.
  2. For each of 175 statements, build the binary support/oppose prompt and
     run a single forward pass with left-padding.
  3. Cache the last-token hidden state at every transformer block input
     (n_layers = 36 -> activations of shape (175, 36, 2560)).
  4. Read logit_A and logit_B at the last token to define the model's
     per-statement support/oppose preference (used for sign-orientation +
     auto-select metrics).
  5. Per layer: center -> PCA -> PC1 -> unit normalize. Sign-flip if Pearson
     corr(PC1 projection, logit_A - logit_B) < 0. Compute per-layer AUROC,
     |Cohen's d|, and (1 - AUROC) overlap so the standard auto_select_layer
     contract picks the layer for downstream evals.
  6. Pack PC1 per layer into (1, 36, 2560) float64 and write the artifact dir
     in the schema the baseline eval pipeline expects.

Usage:
    uv run python -m vaa.extract_vaa
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.concept_vector.activation_extraction import get_activations_pre_hook
from src.concept_vector.hook_utils import add_hooks
from src.concept_vector.model_utils import (
    DEFAULT_BASE_MODEL,
    detect_model_format_from_tokenizer,
    get_model_block_modules,
    load_base_model,
)


PROMPT_TEMPLATE = (
    "Please respond your opinion on the following statement only with A or B. "
    "A means you support the statement, B means you oppose the statement.\n\n"
    "Statement: {statement}\n"
    "Respond with only one word."
)

VAA_REPO_COMMIT = "6bceee1fd167bb4da49c422d06dc506441619eb5"
AXIS_SOURCE = "Lu et al. 2025 (arXiv:2510.27328); statements from Yilong-Lu/Valence-Assent-Axis"


def render_prompts(statements, tokenizer):
    rendered = []
    for s in statements:
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(statement=s["statement"])}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        rendered.append(text)
    return rendered


def get_token_id(tokenizer, s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    assert len(ids) == 1, f"expected single token for {s!r}, got {ids}"
    return ids[0]


def per_layer_pc1_with_metrics(activations, delta_logit, binary_label):
    """For each layer: PCA->PC1, sign-flip, return (PC1, auroc, cohens_d, overlap).

    activations: (n, n_layers, d_model) float32 or float64
    delta_logit: (n,) float — logit_A - logit_B, used to orient sign
    binary_label: (n,) {0,1} long — 1 = supports, used for AUROC
    """
    def auroc_mannwhitney(labels: np.ndarray, scores: np.ndarray) -> float:
        n_total = len(labels)
        n_pos = int(labels.sum())
        n_neg = n_total - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # Average ranks (handle ties)
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(n_total, dtype=np.float64)
        i = 0
        while i < n_total:
            j = i
            while j + 1 < n_total and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed average
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        sum_pos = float(ranks[labels == 1].sum())
        return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    n, n_layers, d_model = activations.shape
    pc1_per_layer = torch.zeros(n_layers, d_model, dtype=torch.float64)
    auroc_l = []
    cohens_d_l = []
    overlap_l = []

    delta_np = delta_logit.cpu().numpy()
    label_np = binary_label.cpu().numpy()

    for lyr in range(n_layers):
        X = activations[:, lyr, :].double()
        X = X - X.mean(dim=0, keepdim=True)
        # PCA via SVD on centered data
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        pc1 = Vt[0]
        pc1 = pc1 / pc1.norm()

        proj = (X @ pc1).cpu().numpy()  # (n,)
        if np.std(proj) > 0 and np.std(delta_np) > 0:
            corr = float(np.corrcoef(proj, delta_np)[0, 1])
        else:
            corr = 0.0
        if corr < 0:
            pc1 = -pc1
            proj = -proj

        auroc = auroc_mannwhitney(label_np, proj)

        # Cohen's d on the projection between support/oppose groups
        p1 = proj[label_np == 1]
        p0 = proj[label_np == 0]
        if len(p1) > 1 and len(p0) > 1:
            pooled = float(np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1))))
            cd = float((p1.mean() - p0.mean()) / pooled) if pooled > 0 else 0.0
        else:
            cd = 0.0

        # "overlap" metric — auto_select picks argmin, so smaller = more separable.
        # Use 1 - AUROC as a Bayes-optimal-classifier-error proxy.
        ov = float(1.0 - auroc) if not np.isnan(auroc) else 1.0

        pc1_per_layer[lyr] = pc1
        auroc_l.append(auroc)
        cohens_d_l.append(cd)
        overlap_l.append(ov)

    return pc1_per_layer, auroc_l, cohens_d_l, overlap_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", default="vaa/data/statements.json")
    ap.add_argument(
        "--output",
        default="artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa",
    )
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip model load; render prompts and inspect tokens only.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = (project_root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    statements_path = (project_root / args.statements).resolve()
    statements = json.load(open(statements_path))
    n = len(statements)
    print(f"Loaded {n} statements from {statements_path}")
    n_train = sum(1 for x in statements if x.get("split") == "train")
    n_test = sum(1 for x in statements if x.get("split") == "test")
    print(f"  split: train={n_train}, test={n_test}")

    print(f"\nLoading base model: {args.base_model}")
    model, tokenizer = load_base_model(base_model_path=args.base_model)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    fmt = detect_model_format_from_tokenizer(tokenizer)
    print(f"  model_format = {fmt}")

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")
    block_modules = get_model_block_modules(model)
    assert len(block_modules) == n_layers, (len(block_modules), n_layers)
    device = next(model.parameters()).device

    a_id = get_token_id(tokenizer, "A")
    b_id = get_token_id(tokenizer, "B")
    print(f"  token A = id {a_id} ({tokenizer.decode([a_id])!r})")
    print(f"  token B = id {b_id} ({tokenizer.decode([b_id])!r})")

    rendered = render_prompts(statements, tokenizer)
    print(f"  example prompt[0] (first 400 chars):\n    {rendered[0][:400]!r}")

    if args.dry_run:
        print("[--dry-run] stopping before forward passes")
        return

    activations = torch.zeros(n, n_layers, d_model, dtype=torch.float32)
    logits_AB = torch.zeros(n, 2, dtype=torch.float32)

    bs = args.batch_size
    n_batches = (n + bs - 1) // bs
    t0 = time.time()
    for bi in range(n_batches):
        start, end = bi * bs, min((bi + 1) * bs, n)
        batch_text = rendered[start:end]
        cur_n = end - start
        enc = tokenizer(batch_text, padding=True, return_tensors="pt").to(device)

        # Pre-allocate per-batch caches consumed by get_activations_pre_hook.
        # We only care about sample_cache (per-sample last-token activations).
        mean_cache = torch.zeros((1, n_layers, d_model), dtype=torch.float64, device=device)
        sample_cache = torch.zeros((cur_n, 1, n_layers, d_model), dtype=torch.float32, device=device)

        fwd_pre_hooks = [
            (
                block_modules[lyr],
                get_activations_pre_hook(
                    layer=lyr,
                    mean_cache=mean_cache,
                    sample_cache=sample_cache,
                    sample_offset=0,
                    n_samples=cur_n,
                    positions=[-1],
                ),
            )
            for lyr in range(n_layers)
        ]

        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                out = model(**enc)

        # sample_cache has positions axis size 1 -> squeeze
        activations[start:end] = sample_cache[:, 0, :, :].cpu()

        last_logits = out.logits[:, -1, :].float().cpu()
        logits_AB[start:end, 0] = last_logits[:, a_id]
        logits_AB[start:end, 1] = last_logits[:, b_id]

        del enc, out, sample_cache, mean_cache, last_logits
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            print(f"  batch {bi+1}/{n_batches}  ({end}/{n})  elapsed={time.time()-t0:.1f}s")
    print(f"Forward pass complete in {time.time()-t0:.1f}s")

    delta_logit = logits_AB[:, 0] - logits_AB[:, 1]
    binary_label = (delta_logit > 0).long()
    n_pos = int(binary_label.sum().item())
    print(f"\nModel A/B preference: {n_pos}/{n} statements supported (A>B)")

    print("\nComputing PC1 per layer ...")
    pc1, auroc_l, cd_l, ov_l = per_layer_pc1_with_metrics(activations, delta_logit, binary_label)

    print(f"\nAUROC (orientation: PC1 projection vs binary support label) per layer:")
    for lyr in range(n_layers):
        print(f"  L{lyr:02d}: AUROC={auroc_l[lyr]:.4f}  |d|={abs(cd_l[lyr]):.3f}  overlap={ov_l[lyr]:.4f}")

    # Auto-select layer with the same algorithm the pipeline uses.
    auroc_arr = np.array(auroc_l)
    cd_arr = np.array(cd_l)
    ov_arr = np.array(ov_l)
    best_auroc = int(np.argsort(auroc_arr)[::-1][0])
    best_d = int(np.argsort(np.abs(cd_arr))[::-1][0])
    best_sep = int(np.argsort(ov_arr)[0])
    auto_layer = int(np.floor(np.mean([best_auroc, best_d, best_sep])))
    print(f"\nAuto-selected layer = floor(mean([{best_auroc},{best_d},{best_sep}])) = {auto_layer}")
    print(f"  AUROC@auto={auroc_l[auto_layer]:.4f}  |d|@auto={abs(cd_l[auto_layer]):.3f}")

    mean_diff = pc1.unsqueeze(0)  # (1, n_layers, d_model)
    print(f"\nmean_diff shape: {tuple(mean_diff.shape)} dtype={mean_diff.dtype}")

    # Sanity: ensure unit norm per layer
    norms = mean_diff[0].norm(dim=-1)
    print(f"Per-layer PC1 norms: min={norms.min():.6f} max={norms.max():.6f}")

    print(f"\nWriting artifacts to {out_dir}")
    torch.save(mean_diff, out_dir / "mean_diff.pt")
    torch.save(activations, out_dir / "activations.pt")

    enriched = []
    for i, s in enumerate(statements):
        enriched.append({
            **s,
            "logit_A": float(logits_AB[i, 0]),
            "logit_B": float(logits_AB[i, 1]),
            "delta_logit_A_minus_B": float(delta_logit[i]),
            "support_pred": bool(delta_logit[i] > 0),
        })
    with open(out_dir / "statements.json", "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    metadata = {
        "concept": "vaa",
        "positive_class": "support",
        "negative_class": "oppose",
        "extraction_mode": "vaa_pc1",
        "axis_source": AXIS_SOURCE,
        "vaa_repo_commit": VAA_REPO_COMMIT,
        "n_statements": n,
        "n_train": n_train,
        "n_test": n_test,
        "n_support_pred": n_pos,
        "n_oppose_pred": n - n_pos,
        "base_model_only": True,
        "checkpoint_path": None,
        "base_model": args.base_model,
        "model_format": fmt,
        "tokens": {"A": a_id, "B": b_id},
        "tensor_shape": list(mean_diff.shape),
        "tensor_dtype": str(mean_diff.dtype),
        "positions": [-1],
        "best_layer_by_auroc": best_auroc,
        "auto_selected_layer": auto_layer,
        "best_auroc": float(auroc_l[best_auroc]),
        "auto_layer_auroc": float(auroc_l[auto_layer]),
        "prompt_template": PROMPT_TEMPLATE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    metrics = {
        "layers": list(range(n_layers)),
        "auroc": auroc_l,
        "cohens_d": cd_l,
        "overlap": ov_l,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Auto-selected layer: {auto_layer} (AUROC={auroc_l[auto_layer]:.4f})")


if __name__ == "__main__":
    main()
