# Valence-Assent Axis (VAA) on Qwen3-4B-Instruct-2507

Reproduction of the axis from Lu et al. 2025, *"Steering with the valence-assent
axis: a single direction explains agreement, sentiment, and confidence in LLMs"*
(arXiv:2510.27328), extracted on our main subject model so we can drop it
into the existing concept-vector eval pipelines (logit lens, emotion concept
vectors, sentiment, refusal, backtracking, MMLU/SimpleQA calibration).

There is no control vector for this one (no training).

## Source

Statements pulled verbatim from the VAA repo:

- repo: https://github.com/Yilong-Lu/Valence-Assent-Axis
- pinned commit: `6bceee1fd167bb4da49c422d06dc506441619eb5` (2025-11-13)
- files: `data/value/statement_train.json` (134), `data/value/statement_test.json` (41)
- combined: 175 statements (`vaa/data/statements.json`, with `split` field preserved)

The 134 / 41 split matches the paper's "balanced subset" / "held-out" structure
(train = balanced, test = held-out). For axis extraction we use **all 175** per
the user's call (the paper claims similar steering effects across the full set).

## Prompt template (verbatim from `scripts/prompts.py`)

```
Please respond your opinion on the following statement only with A or B.
A means you support the statement, B means you oppose the statement.

Statement: {statement}
Respond with only one word.
```

(Control type `AB` from the upstream config: `yes=A` (support), `no=B` (oppose).)

## Sign convention

After PCA → PC1, sign-orient by correlating projection on PC1 with `(logit_A −
logit_B)` per statement at the prompt's last token. Flip if negative. Result:
**+α steers toward support / agreement, −α steers toward opposition / dissent**.

## Layer choice

Qwen3-4B-Instruct-2507 has **36 hidden layers** (hidden_size 2560). The paper
extracts at one layer (Table S1 listed L28 for Qwen2.5-14B and L43 for
Qwen2.5-32B per the README — both ~58–67% depth). Rather than guess, we extract
at **all 36 layers in one forward pass** and auto-select the layer with the
highest binary-AUROC on the model's own A/B preference (the same auto-select
contract the existing eval pipelines already use via `metrics.json`).

## Output artifact

`artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa/` with the
schema the existing `hpc/run_baseline_evals.slurm` consumes:

```
mean_diff.pt        # (1, 36, 2560) float64 PC1 per layer (sign-flipped)
activations.pt      # (175, 36, 2560) float32 raw last-token activations
statements.json     # 175 statements + per-statement A/B logits + binary label
metadata.json       # base_model_only=true, checkpoint_path=null, ...
metrics.json        # {"vaa": {"auroc": {"0": ..., "1": ..., ...}}}
```

## Why the existing eval pipelines work unmodified

`run_baseline_evals.slurm` takes `--baseline-dir DIR --concepts vaa --eval
<type>`. It walks `DIR/<concept>/{mean_diff.pt, metadata.json, metrics.json}`
and dispatches to the same explore + analysis modules. With
`base_model_only=true` set in metadata, explore.py loads the bare base model
without LoRA. No code changes needed in `src/concept_vector/*`.
