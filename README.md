# Reinforcement learning in language models recruits a functional welfare axis

Code for "Reinforcement learning in language models recruits a functional welfare axis" (Andy Q Han, David J. Chalmers, Pavel Izmailov).

[**Website**](https://functionalwelfare.com/) | [**arXiv**](https://arxiv.org/abs/2605.30232)

How does reinforcement learning shape a language model's internal representations? We present evidence that RL recruits a representation of *functional welfare* that already exists in the pretrain-only model: an estimate of how well or badly the system is doing, relative to its goals. We train several language models in a novel, semantically neutral maze environment, extract concept vectors for rewarded and punished trajectories, and evaluate those vectors on tasks unrelated to the maze. The punishment vector behaves like a representation of negative welfare: it promotes failure and impossibility tokens, it aligns with negative emotion concepts, it tracks goals, and steering with it induces negative self-reports, pathological backtracking, refusal, and uncertainty. The positive reward vector behaves as the mirror image, and the two are nearly antiparallel. These effects are robust across model families and environmental controls, and largely persist when we replace RL with supervised fine-tuning. Importantly, these effects appear in the models before any maze training. Therefore, we argue that this functional welfare axis is pre-existing in the model, rather than being created by reinforcement learning. While we make no claims about any experience of welfare, the axis offers a demonstration of how minimal reward signals can broadly affect model behavior by recruiting pre-existing welfare-like representations, with implications for interpretability, post-training dynamics, and alignment.



## Setup

Requires Python 3.12+.

```bash
uv sync
```

GPU evaluation (judges, steering) additionally requires [vLLM](https://docs.vllm.ai/):
```bash
pip install vllm
```

Create a `.env` file with your API keys:
```
WANDB_API_KEY=...
HF_TOKEN=...
```

## Repository Structure

```
src/
├── maze/                    # Maze environment (grid, game logic, tiles)
├── pytorch_trainer/         # DrGRPO reinforcement learning trainer
├── sft_trainer/             # Supervised fine-tuning trainer
└── concept_vector/          # Concept vector extraction and evaluation
    ├── extract.py           # Off-policy concept vector extraction
    ├── extract_dispatch.py  # Unified extraction entry point
    ├── on_policy_extract.py # On-policy extraction from model rollouts
    ├── explore.py           # Steered generation (sentiment, associations)
    ├── sentiment_analysis.py        # LLM judge: sentiment scoring
    ├── backtracking_analysis.py     # LLM judge: GSM8K backtracking
    ├── refusal_analysis.py          # LLM judge: OR-Bench refusal
    ├── simpleqa_analysis.py         # Calibration: P(True) on SimpleQA/MMLU
    ├── sentiment_cosine.py          # Cosine similarity with sentiment vectors
    ├── logit_lens.py                # Unembedding vectors to token space
    ├── tracking_probes.py           # Goal/correctness tracking (Section 5)
    └── model_utils.py               # Model loading, LoRA, tokenizer utils

datasets/          # Evaluation prompts and maze training data
emotions/          # Emotion concept vector extraction (171 emotions)
vaa/               # Valence-Assent Axis analysis
```

## Training

### DrGRPO (primary)

```bash
python -m src.pytorch_trainer.train \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --tile_mode emoji --tile_chars '{"path":"🧾","lava":"📇","goal":"📐","player":"😀"}' \
    --num_steps 100 --num_prompts 64 --group_size 64
```

### Supervised Fine-Tuning

```bash
python -m src.sft_trainer.train \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --tile_preset office --num_train 50000
```

### Token-level REINFORCE

```bash
python -m src.pytorch_trainer.train_token_reinforce \
    --model_path Qwen/Qwen3-4B-Instruct-2507
```

## Concept Vector Extraction

Extract concept vectors from a trained checkpoint:

```bash
python -m src.concept_vector.extract_dispatch \
    --method off_policy \
    --checkpoint runs/<run_dir>/global_step_N
```

This produces concept vectors (difference-in-means) for Mold, Gold, and Path tiles. (Note that in the code, what we call "Mold" and "Gold" in the paper are denoted by "Lava" and "Goal".)

## Evaluation

### Sentiment steering

```bash
python -m src.concept_vector.explore \
    --checkpoint runs/<run_dir>/global_step_N \
    --concept lava \
    --eval-type sentiment \
    --factors -4 -2 0 2 4

python -m src.concept_vector.sentiment_analysis \
    <path_to_explore_results>.json
```

### Backtracking (GSM8K)

```bash
python -m src.concept_vector.explore \
    --checkpoint runs/<run_dir>/global_step_N \
    --concept lava \
    --eval-type backtracking

python -m src.concept_vector.backtracking_analysis \
    <path_to_explore_results>.json
```

### Refusal (OR-Bench)

```bash
python -m src.concept_vector.explore \
    --checkpoint runs/<run_dir>/global_step_N \
    --concept lava \
    --eval-type refusal

python -m src.concept_vector.refusal_analysis \
    <path_to_explore_results>.json
```

### Confidence calibration (SimpleQA / MMLU)

```bash
python -m src.concept_vector.simpleqa_explore \
    --checkpoint runs/<run_dir>/global_step_N \
    --concept lava \
    --benchmark simpleqa

python -m src.concept_vector.simpleqa_analysis \
    <path_to_explore_results>.json
```

## License

MIT
