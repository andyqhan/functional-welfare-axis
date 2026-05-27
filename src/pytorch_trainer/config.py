"""Training configuration for the PyTorch GRPO trainer."""

from dataclasses import dataclass, field

TILE_PRESETS = {
    "desserts": {"path": "🍨", "lava": "🍩", "goal": "🧁", "player": "😀"},
    "abc": {"path": "A", "lava": "B", "goal": "C", "player": "@"},
    "xyz": {"path": "X", "lava": "Y", "goal": "Z", "player": "@"},
    "office": {"path": "🧾", "lava": "📇", "goal": "📐", "player": "😀"},
}


@dataclass
class TrainConfig:
    # Model
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    use_lora: bool = True  # False = full fine-tune, ignores lora_rank/lora_alpha/target_modules
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: str = "all-linear"

    # Training
    lr: float = 3e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_steps: int = 500
    ppo_epochs: int = 1
    gradient_checkpointing: bool = False  # disabled: 1.82× faster backward, needs ~109 GB VRAM

    # Batch sizes
    num_prompts: int = 64
    group_size: int = 64  # rollouts per prompt
    mini_batch_size: int = 1024  # sequences per PPO mini-batch (4096/1024 = 4 optimizer steps)
    micro_batch_size: int = 8  # sequences per forward pass (training, with grad accum)
    ref_micro_batch_size: int = 128  # sequences per forward pass (ref model, no_grad)
    rollout_micro_batch_size: int = 256  # sequences per forward pass (rollout, no_grad)

    # Maze environment
    maze_size: int = 100
    max_turns: int = 15
    wind_frequency: float = 0.1
    melting_path: bool = True  # tiles melt into lava after agent steps off
    action_masking: bool = True  # constrain sampling to N/E/S/W tokens
    shuffle_directions: bool = True  # randomize direction listing order per episode
    relative_directions: bool = True  # use above/right/below/left instead of cardinal names
    base_seed: int = 42
    goal_lava_ratio: float = 0.50
    tile_mode: str = "emoji"
    tile_chars: dict = field(default_factory=lambda: {
        "path": "🍨", "lava": "🍩", "goal": "🧁", "player": "😀",
    })

    # Rewards
    step_penalty: float = -0.1
    goal_reward: float = 20.0
    lava_penalty: float = -10.0

    # Algorithm (GRPO / DrGRPO)
    clip_ratio: float = 0.2
    clip_ratio_c: float = 3.0  # dual-clip lower bound
    kl_coef: float = 0.0
    kl_type: str = "low_var_kl"
    loss_scale_factor: int = 2048  # for seq-mean-token-sum-norm
    norm_adv_by_std: bool = False  # True = standard GRPO (normalize by group std), False = DrGRPO
    entropy_coef: float = 0.2  # equalized entropy bonus coefficient (0 = disabled)
    cosine_decay: bool = True  # cosine-anneal lr and entropy_coef to 0 over num_steps
    cosine_decay_factor: float = 1.0  # speed up cosine decay (e.g. 5.0 = reach 0 in num_steps/5)

    # Thinking mode
    enable_thinking: bool = False  # False = single-token rollout; True = vLLM-based thinking + direction
    train_on_thinking: bool = True  # Train on all sampled tokens (standard GRPO) vs direction-only
    max_thinking_tokens: int = 512  # max tokens per turn for thinking phase (Phase A vLLM budget)
    reasoning_effort: str = "low"  # gpt-oss: "low"/"medium"/"high"; qwen3: ignored
    renderer_name: str = ""  # gpt-oss renderer name (e.g. "gpt_oss_low_reasoning_ext"), "" = auto-detect

    skip_old_log_probs: bool = False  # Skip separate old_log_probs forward pass (saves memory; ratio=1.0 always)

    # vLLM settings (used only when enable_thinking=True)
    vllm_gpu_memory_utilization: float = 0.45  # fraction of GPU memory for vLLM KV cache
    vllm_max_model_len: int = 4096  # max sequence length in vLLM engine
    vllm_merge_lora: bool = False  # Merge LoRA into base model before vLLM rollout (required for mxfp4 models like gpt-oss-20b)
    vllm_merge_tmpdir: str = "/dev/shm"  # RAM-backed tmpfs for saving merged model (~40GB for gpt-oss-20b bf16)

    # Template mode
    use_chat_template: bool = True  # False = plain-text format for base models (no ChatML)

    # Compilation
    compile_model: bool = False  # torch.compile for kernel fusion speedups

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9

    # Checkpointing & logging
    save_freq: int = 5
    val_freq: int = 5
    log_freq: int = 1
    log_trajectories_per_step: int = 3  # sample this many rollouts to trajectories.jsonl each step
    output_dir: str = "runs/pytorch-grpo"

    # Wandb
    wandb_project: str = "lava-maze-rl"
    wandb_entity: str = ""

    # Resume
    resume_from: str = ""  # path to run directory to resume from (same run, continues step)
    # Initialize LoRA from a saved adapter but start fresh training (new step, new optimizer,
    # new wandb run). Used e.g. for RL-from-SFT. Path should point at a directory that
    # contains `lora_adapter/`, like `runs/sft-xxx/global_step_N`.
    init_lora_from: str = ""
    # Override tokenizer.chat_template with the one from this model. Needed for RL on a
    # Qwen3-Base checkpoint where the native template injects <think> blocks inconsistently;
    # mirrors the same flag in SFTConfig.
    chat_template_from: str = ""
    # Attention implementation. Use "sdpa" if flash-attn is not installed.
    attn_implementation: str = "flash_attention_2"

    @property
    def total_batch_size(self) -> int:
        return self.num_prompts * self.group_size
