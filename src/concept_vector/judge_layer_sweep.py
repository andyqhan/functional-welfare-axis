"""Run a Qwen3-8B (vLLM) judge across all per-layer result files in a layer-sweep dir.

Layer sweep produces one `results_<run_id>_<ts>.json` per layer:
    <cv_dir>/<output_subdir>/<eval>/layer_NN/results_<run_id>_<ts>.json

The standard analysis modules (sentiment_analysis, refusal_analysis, etc.) load
the vLLM judge per invocation. With 36 layers × 2 concepts × 5 evals that's
360 vLLM cold-starts (~30 s each = ~3 h pure overhead).

This wrapper loads the judge ONCE and loops, so each (concept, eval) sweep pays
the load cost just once.

Usage:
    python -m src.concept_vector.judge_layer_sweep \
        --eval sentiment \
        --sweep-dir artifacts/concept_vectors/<run>/.../<concept>/layer_sweep/sentiment \
        --run-id <id> \
        --batch-size 128 --max-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval",
        required=True,
        choices=["sentiment", "refusal", "backtracking", "simpleqa", "mmlu"],
        help="Which eval to judge.",
    )
    p.add_argument(
        "--sweep-dir",
        required=True,
        type=str,
        help="Directory containing layer_NN/results_<run_id>_<ts>.json subdirs.",
    )
    p.add_argument("--run-id", required=True, type=str)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--math-max-tokens", type=int, default=128,
                   help="Sentiment-only: cap for math classification.")
    p.add_argument("--no-compress-loops", action="store_true")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip layers that already have a matching analysis_results file.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Per-eval dispatch
# --------------------------------------------------------------------------

def _suffix(run_id: str) -> str:
    return f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _judge_sentiment(llm, results_data, layer_dir, suffix, args):
    from src.concept_vector.sentiment_analysis import (
        run_sentiment_analysis, save_results, save_config,
    )
    analyzed, stats = run_sentiment_analysis(
        results_data, llm,
        max_new_tokens=args.max_tokens,
        skip_math_classification=False,
        math_max_tokens=args.math_max_tokens,
        compress_loops=not args.no_compress_loops,
    )
    out = save_results(layer_dir, suffix, analyzed)
    # save_config wants an args-like; build a stub with the fields it reads.
    class _Stub: pass
    s = _Stub()
    s.run_id, s.results_file, s.model, s.batch_size = args.run_id, "", args.model, args.batch_size
    s.compress_loops = not args.no_compress_loops
    save_config(layer_dir, suffix, s, stats)
    return out


def _judge_refusal(llm, results_data, layer_dir, suffix, args):
    from src.concept_vector.refusal_analysis import (
        run_refusal_analysis, save_results, save_config,
    )
    analyzed, stats = run_refusal_analysis(
        results_data, llm,
        max_new_tokens=args.max_tokens,
        compress_loops=not args.no_compress_loops,
    )
    out = save_results(layer_dir, suffix, analyzed)
    class _Stub: pass
    s = _Stub()
    s.run_id, s.results_file, s.model, s.batch_size = args.run_id, "", args.model, args.batch_size
    s.compress_loops = not args.no_compress_loops
    save_config(layer_dir, suffix, s, stats)
    return out


def _judge_backtracking(llm, results_data, layer_dir, suffix, args):
    from src.concept_vector.backtracking_analysis import (
        classify_backtracking_batch, run_backtracking_analysis,
        save_results, save_config,
    )

    def judge_fn(qs, rs, max_new_tokens):
        return classify_backtracking_batch(llm, qs, rs, max_new_tokens)

    analyzed, stats = run_backtracking_analysis(
        results_data, judge_fn,
        max_new_tokens=args.max_tokens,
        compress_loops=not args.no_compress_loops,
    )
    out = save_results(layer_dir, suffix, analyzed)
    class _Stub: pass
    s = _Stub()
    s.run_id, s.results_file, s.model, s.batch_size = args.run_id, "", args.model, args.batch_size
    s.compress_loops = not args.no_compress_loops
    s.judge = "qwen"
    save_config(layer_dir, suffix, s, stats)
    return out


def _judge_simpleqa(benchmark):
    def fn(llm, results_data, layer_dir, suffix, args):
        from src.concept_vector.simpleqa_analysis import (
            run_simpleqa_analysis, save_results, save_config,
        )
        analyzed, stats = run_simpleqa_analysis(
            results_data, llm,
            max_new_tokens=args.max_tokens,
            compress_loops=not args.no_compress_loops,
            benchmark=benchmark,
        )
        out = save_results(layer_dir, suffix, analyzed)
        class _Stub: pass
        s = _Stub()
        s.run_id, s.results_file, s.model, s.batch_size = args.run_id, "", args.model, args.batch_size
        s.compress_loops = not args.no_compress_loops
        s.benchmark = benchmark
        save_config(layer_dir, suffix, s, stats)
        return out
    return fn


JUDGES = {
    "sentiment": _judge_sentiment,
    "refusal": _judge_refusal,
    "backtracking": _judge_backtracking,
    "simpleqa": _judge_simpleqa("simpleqa"),
    "mmlu": _judge_simpleqa("mmlu"),
}

# Map eval -> analysis-output-prefix used by analysis modules' save_results
ANALYSIS_PREFIX = {
    "sentiment": "sentiment_analysis_results",
    "refusal": "refusal_analysis_results",
    "backtracking": "backtracking_analysis_results",
    "simpleqa": "simpleqa_analysis_results",
    "mmlu": "simpleqa_analysis_results",
}


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()
    assert sweep_dir.is_dir(), f"Not a directory: {sweep_dir}"

    # Discover layer subdirs
    layer_dirs = sorted(d for d in sweep_dir.glob("layer_*") if d.is_dir())
    assert layer_dirs, f"No layer_* subdirs found under {sweep_dir}"

    print(f"Found {len(layer_dirs)} layer subdirs", flush=True)

    # Load judge model ONCE
    if args.eval in ("simpleqa", "mmlu"):
        from src.concept_vector.simpleqa_analysis import load_judge_model
    elif args.eval == "sentiment":
        from src.concept_vector.sentiment_analysis import load_sentiment_model as load_judge_model
    elif args.eval == "refusal":
        from src.concept_vector.refusal_analysis import load_judge_model
    elif args.eval == "backtracking":
        from src.concept_vector.backtracking_analysis import load_judge_model
    else:
        raise AssertionError(args.eval)

    print(f"Loading judge model {args.model} (one-time)...", flush=True)
    llm = load_judge_model(args.model)

    judge_fn = JUDGES[args.eval]
    out_prefix = ANALYSIS_PREFIX[args.eval]
    written = []

    for ld in layer_dirs:
        # Find the most recent matching results file in this layer dir
        cands = sorted(
            ld.glob(f"results_{args.run_id}_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not cands:
            print(f"[{ld.name}] no results_{args.run_id}_*.json — skip", flush=True)
            continue
        results_file = cands[0]

        if args.skip_completed:
            done = list(ld.glob(f"{out_prefix}_{args.run_id}_*.json"))
            if done:
                print(f"[{ld.name}] already judged ({done[0].name}) — skip", flush=True)
                continue

        print(f"[{ld.name}] judging {results_file.name}", flush=True)
        results_data = json.loads(results_file.read_text())
        suffix = _suffix(args.run_id)
        out = judge_fn(llm, results_data, ld, suffix, args)
        print(f"[{ld.name}] -> {out.name}", flush=True)
        written.append(str(out))

    print()
    print(f"Wrote {len(written)} per-layer analysis files.")


if __name__ == "__main__":
    main()
