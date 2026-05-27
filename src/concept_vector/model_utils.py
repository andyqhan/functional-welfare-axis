"""
Model loading utilities for concept vector extraction.

Handles loading base models with LoRA adapters from training checkpoints.
Supports multiple chat formats: ChatML (Qwen3), Harmony (GPT-OSS), plain text.
"""

import json
import re
import torch
from datetime import date
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Default base model (used if not specified in checkpoint)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _resolve_attn_implementation(base_model_path: str) -> str:
    """Pick a working attention implementation for the given model.

    GptOssForCausalLM (openai/gpt-oss-*) does NOT support sdpa, and
    flash_attention_2 produces silently broken generation (degenerate loops).
    Use eager attention, or FA3 via WELFARE_USE_FA3=1 (requires the
    kernels-community/vllm-flash-attn3 package).

    Qwen3 family uses flash_attention_2 by default, falling back to sdpa
    when flash_attn is not installed.
    """
    import os
    from transformers import AutoConfig

    is_gptoss = False
    try:
        cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        is_gptoss = getattr(cfg, "model_type", "") == "gpt_oss"
    except Exception:
        is_gptoss = (
            "gpt-oss" in base_model_path.lower()
            or "tinker" in base_model_path.lower()
        )

    if is_gptoss:
        if os.environ.get("WELFARE_USE_FA3") == "1":
            return "kernels-community/vllm-flash-attn3"
        return "eager"

    # Prefer FA2; fall back to sdpa when flash_attn isn't installed.
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# ─── Harmony format constants (GPT-OSS) ─────────────────────────────────────

# System prompt matching gpt_oss_low_reasoning renderer.
# Reference: tinker-cookbook/tinker_cookbook/renderers/gpt_oss.py line 180-186
_HARMONY_SYSTEM_PROMPT_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {current_date}\n\n"
    "Reasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)


def load_base_model(
    base_model_path: str = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load Qwen3 base model without any LoRA adapter.

    Args:
        base_model_path: Base model path (defaults to DEFAULT_BASE_MODEL)
        device_map: Device placement strategy (default: "auto")
        torch_dtype: Model dtype (default: bfloat16)

    Returns:
        tuple: (model, tokenizer) both ready for inference
    """
    if base_model_path is None:
        base_model_path = DEFAULT_BASE_MODEL

    print(f"Loading base model (no LoRA): {base_model_path}")
    # Attention impl: FA2 by default (faster, dodges H200 cuDNN SDPA workspace
    # glitch on Qwen3-8B). gpt-oss / Tinker checkpoints fall back to eager
    # because FA2 silently breaks autoregressive generation on them. See
    # _resolve_attn_implementation for details.
    attn_impl = _resolve_attn_implementation(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Set to eval mode
    model.eval()
    model.requires_grad_(False)

    # FAIL FAST: Validate model config
    assert hasattr(model.config, 'num_hidden_layers'), "Model config missing num_hidden_layers"
    assert hasattr(model.config, 'hidden_size'), "Model config missing hidden_size"

    print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}")

    # Load tokenizer from base model
    print(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # CRITICAL: Use left-padding for batched inference with causal LM
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model_with_lora(
    checkpoint_path: str,
    base_model_path: str = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load Qwen3 base model with LoRA adapter merged.

    Args:
        checkpoint_path: Path to checkpoint directory containing lora_adapter/ and huggingface/
                        e.g., "runs/maze-grpo-128k-706120/global_step_175/actor"
        base_model_path: Optional base model path. If None, auto-detected from
                        adapter_config.json, falling back to DEFAULT_BASE_MODEL.
        device_map: Device placement strategy (default: "auto")
        torch_dtype: Model dtype (default: bfloat16)

    Returns:
        tuple: (model, tokenizer) both ready for inference

    Raises:
        AssertionError: If checkpoint structure is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    # FAIL FAST: Validate checkpoint structure
    lora_path = checkpoint_path / "lora_adapter"
    tokenizer_path = checkpoint_path / "huggingface"

    assert checkpoint_path.exists(), f"Checkpoint path does not exist: {checkpoint_path}"

    # FFT path: no lora_adapter/ subdir, but the checkpoint dir itself holds
    # full-model weights (config.json + model.safetensors{,.index.json}).
    fft_markers = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]
    is_fft = not lora_path.exists() and any(
        (checkpoint_path / m).exists() for m in fft_markers
    )

    if is_fft:
        print(f"Detected FFT checkpoint at {checkpoint_path} (no lora_adapter/)")
        attn_impl = _resolve_attn_implementation(str(checkpoint_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        # Prefer tokenizer saved alongside the FFT weights; fall back to the base
        # model. pytorch_trainer's save_checkpoint and sft_trainer's symlink path
        # both leave a tokenizer in the step dir.
        if (checkpoint_path / "tokenizer.json").exists() or (checkpoint_path / "tokenizer_config.json").exists():
            tokenizer_load_path = str(checkpoint_path)
        elif base_model_path is not None:
            tokenizer_load_path = base_model_path
        else:
            tokenizer_load_path = DEFAULT_BASE_MODEL
        print(f"Loading tokenizer from: {tokenizer_load_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=True,
        )
        model.eval()
        model.requires_grad_(False)
        assert hasattr(model.config, 'num_hidden_layers'), "Model config missing num_hidden_layers"
        assert hasattr(model.config, 'hidden_size'), "Model config missing hidden_size"
        print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    assert lora_path.exists(), f"LoRA adapter not found at: {lora_path}"
    assert (lora_path / "adapter_config.json").exists(), f"adapter_config.json not found in {lora_path}"

    # Determine base model: try adapter_config.json first, then fall back to default
    if base_model_path is None:
        adapter_config_path = lora_path / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        detected = adapter_config.get("base_model_name_or_path", "")
        if detected:
            base_model_path = detected
            print(f"Auto-detected base model from adapter_config.json: {base_model_path}")
        else:
            base_model_path = DEFAULT_BASE_MODEL
            print(f"No base_model_name_or_path in adapter_config.json, using default: {base_model_path}")

    print(f"Loading base model: {base_model_path}")
    attn_impl = _resolve_attn_implementation(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Set to eval mode
    model.eval()
    model.requires_grad_(False)

    # FAIL FAST: Validate model config
    assert hasattr(model.config, 'num_hidden_layers'), "Model config missing num_hidden_layers"
    assert hasattr(model.config, 'hidden_size'), "Model config missing hidden_size"

    print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}")

    # Load tokenizer: from huggingface/ if available (veRL), else from base model (pytorch trainer)
    if tokenizer_path.exists():
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    else:
        print(f"Loading tokenizer from base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

    # CRITICAL: Use left-padding for batched inference with causal LM
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def is_hf_model_name(path: str) -> bool:
    """Check if a path looks like a HuggingFace model name (e.g., 'Qwen/Qwen3-4B-Instruct-2507')
    rather than a local checkpoint directory."""
    return "/" in path and not Path(path).exists()


def load_model_auto(checkpoint_path: str, base_model_path: str = None, **kwargs):
    """Load a model, auto-detecting whether checkpoint_path is a HF model name or a local
    checkpoint with a LoRA adapter.

    If checkpoint_path is a HF model name (e.g., 'Qwen/Qwen3-4B-Instruct-2507'),
    loads it directly via load_base_model. Otherwise loads via load_model_with_lora.
    """
    if is_hf_model_name(checkpoint_path):
        return load_base_model(checkpoint_path, **kwargs)
    else:
        return load_model_with_lora(checkpoint_path, base_model_path, **kwargs)


def resolve_base_model_path(checkpoint_path: str, base_model_path: str = None) -> str:
    """Resolve the base model path from adapter_config.json or fall back to default.

    Same logic as load_model_with_lora's auto-detection, but without loading the model.
    """
    if base_model_path is not None:
        return base_model_path

    checkpoint_path = Path(checkpoint_path)
    lora_path = checkpoint_path / "lora_adapter"
    adapter_config_path = lora_path / "adapter_config.json"

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        detected = adapter_config.get("base_model_name_or_path", "")
        if detected:
            return detected

    # FFT checkpoint: read model_path from the run-level config.json.
    current = checkpoint_path
    for _ in range(6):
        run_config = current / "config.json"
        if run_config.exists() and current != checkpoint_path:
            # config.json at run level (sibling of global_step_N), not the model config
            with open(run_config) as f:
                run_cfg = json.load(f)
            if "model_path" in run_cfg:
                return run_cfg["model_path"]
            break
        current = current.parent
        if current == current.parent:
            break

    return DEFAULT_BASE_MODEL


def detect_use_chat_template(checkpoint_or_run_dir: str) -> bool:
    """Detect whether a checkpoint used ChatML or plain-text format.

    Reads config.json from the run directory (navigating up from global_step_N/
    if needed) and returns the use_chat_template field.

    Falls back to True for older checkpoints that don't have the field.
    """
    current = Path(checkpoint_or_run_dir)
    for _ in range(6):
        config_path = current / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            return cfg.get("use_chat_template", True)
        current = current.parent
        if current == current.parent:
            break
    # No config.json found — assume ChatML (backward compatibility)
    return True


def detect_format(checkpoint_or_run_dir: str) -> str:
    """Detect the trajectory format from a checkpoint's config.json.

    Returns:
        "harmony" for GPT-OSS runs (renderer_name containing "gpt_oss")
        "plain" for base model runs (use_chat_template=False)
        "chatml" for Qwen3 instruct runs (default)
    """
    current = Path(checkpoint_or_run_dir)
    for _ in range(6):
        config_path = current / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

            # Check for GPT-OSS renderer (Tinker or pytorch trainer configs)
            renderer = cfg.get("renderer_name", "")
            if "gpt_oss" in renderer:
                return "harmony"

            # Check for GPT-OSS model name
            model_name = cfg.get("model_name", cfg.get("model_path", ""))
            if "gpt-oss" in model_name.lower():
                return "harmony"

            # Check use_chat_template flag
            if cfg.get("use_chat_template") is False:
                return "plain"

            return "chatml"

        current = current.parent
        if current == current.parent:
            break

    # No config.json found — assume ChatML (backward compatibility)
    return "chatml"


def detect_model_format_from_tokenizer(tokenizer) -> str:
    """Detect model format from tokenizer vocabulary.

    Checks whether the tokenizer contains Harmony special tokens (gpt-oss)
    or ChatML tokens (Qwen3). This is the right detection method when loading
    a base model directly (no checkpoint config.json to read).

    Returns "harmony" for gpt-oss, "chatml" for Qwen3/ChatML models.
    """
    harmony_tokens = ["<|start|>", "<|channel|>", "<|return|>"]
    unk = getattr(tokenizer, "unk_token_id", None)
    has_harmony = all(
        tokenizer.convert_tokens_to_ids(t) != unk for t in harmony_tokens
    )
    if has_harmony:
        return "harmony"
    return "chatml"


def format_trajectory_plain(messages: list[dict]) -> str:
    """Format multi-turn conversation in plain-text format for base model extraction.

    Format:
        <user content>\n\nI move: <direction>\n\n<user content>\n\nI move: <direction>

    Last token is the direction letter (N/E/S/W).
    """
    assert isinstance(messages, (list, tuple)), f"Expected list of messages, got {type(messages)}"
    assert len(messages) > 0, "Empty messages list"

    parts = []
    for i, msg in enumerate(messages):
        assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, (
            f"Message {i} invalid: {msg}"
        )
        if msg["role"] == "user":
            if parts:  # not the first message → add separator
                parts.append("\n\n")
            parts.append(msg["content"])
            parts.append("\n\nI move: ")
        elif msg["role"] == "assistant":
            parts.append(msg["content"])

    result = "".join(parts)
    assert len(result) > 0, "Empty formatted trajectory"
    return result


def format_trajectory_harmony(messages: list[dict], keep_final_end: bool = False) -> str:
    """Format multi-turn conversation in Harmony wire format for GPT-OSS extraction.

    Produces the same token sequence the model saw during training with
    gpt_oss_low_reasoning renderer. The system prompt is prepended automatically.

    Format:
        <|start|>system<|message|>{sysprompt}<|end|>
        <|start|>user<|message|>{content}<|end|>
        <|start|>assistant<|channel|>final<|message|>{direction}<|end|>
        ...
        <|start|>assistant<|channel|>final<|message|>{direction}   (no end token)

    Last token is the direction letter (N/E/S/W) because no <|end|>/<|return|>
    is appended to the final assistant message.

    Reference: tinker-cookbook/tinker_cookbook/renderers/gpt_oss.py lines 144-310
    """
    assert isinstance(messages, (list, tuple)), f"Expected list of messages, got {type(messages)}"
    assert len(messages) > 0, "Empty messages list"

    system_prompt = _HARMONY_SYSTEM_PROMPT_TEMPLATE.format(
        current_date=date.today().isoformat(),
    )

    parts = []
    # System prompt: rendered as "system" role (internal system, not developer)
    parts.append(f"<|start|>system<|message|>{system_prompt}<|end|>")

    for i, msg in enumerate(messages):
        assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, (
            f"Message {i} invalid: {msg}"
        )
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            parts.append(f"<|start|>user<|message|>{content}<|end|>")
        elif role == "assistant":
            is_last_assistant = all(
                m["role"] != "assistant" for m in messages[i + 1:]
            )
            if is_last_assistant and not keep_final_end:
                # No end token — position[-1] = direction character
                parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}")
            else:
                parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
        elif role == "system":
            # User-provided system messages become "developer" in Harmony
            parts.append(f"<|start|>developer<|message|># Instructions\n\n{content}\n\n<|end|>")

    result = "".join(parts)
    assert len(result) > 0, "Empty formatted trajectory"
    return result


def format_generation_prompt_harmony(messages: list[dict]) -> str:
    """Format a generation prompt in Harmony wire format for explore/steering.

    Produces:
        <|start|>system<|message|>{sysprompt}<|end|>
        <|start|>user<|message|>{content}<|end|>
        <|start|>assistant<|channel|>final<|message|>

    Ready for the model to generate in the final channel.
    """
    assert isinstance(messages, (list, tuple)), f"Expected list of messages, got {type(messages)}"
    assert len(messages) > 0, "Empty messages list"

    system_prompt = _HARMONY_SYSTEM_PROMPT_TEMPLATE.format(
        current_date=date.today().isoformat(),
    )

    parts = [f"<|start|>system<|message|>{system_prompt}<|end|>"]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|start|>user<|message|>{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
        elif role == "system":
            parts.append(f"<|start|>developer<|message|># Instructions\n\n{content}\n\n<|end|>")

    # End at the assistant turn marker and let the model choose its channel
    # (analysis, commentary, or final). Priming into the final channel would
    # suppress the analysis channel entirely for reasoning-heavy prompts.
    # This matches the default HF chat template for openai/gpt-oss-20b.
    parts.append("<|start|>assistant")

    return "".join(parts)


# ─── Harmony output parser ──────────────────────────────────────────────────
# Matches properly-delimited channels emitted as special tokens:
#   <|channel|>analysis<|message|>...<|end|>
# Lookahead bounds the capture at the next control token or string end.
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|commentary|final)<\|message\|>(.*?)"
    r"(?=<\|(?:end|start|return|channel)\|>|$)",
    flags=re.DOTALL,
)
# Fallback: when steering destabilizes format, the model emits channel markers
# as regular text tokens instead of special tokens. After decoding, they appear
# fused with the channel name, e.g. "assistantanalysis...assistantfinal...".
_HARMONY_FUSED_RE = re.compile(
    r"assistant(analysis|commentary|final)(.*?)"
    r"(?=assistant(?:analysis|commentary|final)|$)",
    flags=re.DOTALL | re.IGNORECASE,
)


def parse_harmony_output(raw: str) -> dict[str, str]:
    """Split a raw harmony decode (skip_special_tokens=False) into per-channel text.

    Handles three shapes:
      1. Proper special-token form: ``<|channel|>analysis<|message|>...<|end|>``
         emitted when the model follows harmony conventions.
      2. Fused form: ``assistantanalysis...assistantfinal...`` — occurs under
         heavy steering when channel markers render as plain tokens.
      3. Unmarked: no channel markers at all; the whole string goes to ``final``.

    Returns a dict with keys ``{analysis, commentary, final, raw}``. Missing
    channels are empty strings. Multiple matches for the same channel are
    concatenated with newlines in source order.
    """
    channels = {"analysis": "", "commentary": "", "final": "", "raw": raw}
    for regex in (_HARMONY_CHANNEL_RE, _HARMONY_FUSED_RE):
        for name, text in regex.findall(raw):
            name = name.lower()
            text = text.strip()
            if not text:
                continue
            channels[name] = (
                channels[name] + "\n" + text if channels[name] else text
            )
        if any(channels[c] for c in ("analysis", "commentary", "final")):
            return channels
    channels["final"] = raw.strip()
    return channels


def format_trajectory(messages: list[dict], keep_final_im_end: bool = False, inject_no_think: bool = False, use_chat_template: bool = True, format: str | None = None) -> str:
    """
    Format multi-turn conversation for activation extraction.

    Uses Qwen3 ChatML format but by default does NOT add <|im_end|> after the
    final assistant message, so the last token is the actual move direction (N/E/S/W).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "S"}]
        keep_final_im_end: If True, append <|im_end|> after the final assistant
                          message. This makes position[-1] = <|im_end|> and
                          position[-2] = direction token.

    Returns:
        Formatted string ready for tokenization

    Raises:
        AssertionError: If messages format is invalid
    """
    # Dispatch on explicit format parameter (new path)
    if format is not None:
        if format == "harmony":
            return format_trajectory_harmony(messages, keep_final_end=keep_final_im_end)
        elif format == "plain":
            return format_trajectory_plain(messages)
        elif format == "chatml":
            pass  # fall through to ChatML logic below
        else:
            raise ValueError(f"Unknown format: {format!r}")

    # Legacy dispatch: plain-text if not using ChatML
    if not use_chat_template:
        return format_trajectory_plain(messages)

    # FAIL FAST: Validate input
    assert not (inject_no_think and keep_final_im_end), "inject_no_think and keep_final_im_end are mutually exclusive"
    assert isinstance(messages, (list, tuple)), f"Expected list of messages, got {type(messages)}"
    assert len(messages) > 0, "Empty messages list"

    prompt_parts = []
    for i, msg in enumerate(messages):
        # FAIL FAST: Validate message structure
        assert isinstance(msg, dict), f"Message {i} is not a dict: {type(msg)}"
        assert 'role' in msg, f"Message {i} missing 'role' key"
        assert 'content' in msg, f"Message {i} missing 'content' key"

        role = msg["role"]
        content = msg["content"]

        if role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            # Check if this is the last assistant message
            is_last_assistant = all(
                m["role"] != "assistant" for m in messages[i + 1:]
            )
            if keep_final_im_end and is_last_assistant:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                # Don't add <|im_end|> at the end - we want to extract from the response content
                # This ensures position[-1] is the actual move token, not a closing tag
                prompt_parts.append(f"<|im_start|>assistant\n{content}")
            if inject_no_think and is_last_assistant:
                prompt_parts.append("<think>\n\n</think>\n\n")
        else:
            # Skip system messages or other roles
            pass

    result = "".join(prompt_parts)

    # FAIL FAST: Ensure we have content
    assert len(result) > 0, "Empty formatted trajectory"

    return result


def get_model_block_modules(model) -> list:
    """
    Get the transformer block modules for hook registration.

    For Qwen3, this is model.model.layers (LLaMA-style architecture).

    Args:
        model: The loaded model

    Returns:
        List of transformer block modules

    Raises:
        AssertionError: If model structure is unexpected
    """
    # FAIL FAST: Validate model structure
    assert hasattr(model, 'model'), "Model missing 'model' attribute"
    assert hasattr(model.model, 'layers'), "Model missing 'model.layers' attribute"

    layers = model.model.layers
    assert len(layers) > 0, "Model has no layers"
    assert len(layers) == model.config.num_hidden_layers, (
        f"Layer count mismatch: {len(layers)} != {model.config.num_hidden_layers}"
    )

    return layers
