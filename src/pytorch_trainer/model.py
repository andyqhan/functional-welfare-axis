"""Model loading: actor (with LoRA), reference (frozen), and tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from .config import TrainConfig


def load_actor_model(
    config: TrainConfig,
    device: torch.device,
    lora_adapter_path: str | None = None,
) -> torch.nn.Module:
    """Load the actor model with LoRA and flash attention 2.

    Args:
        config: training configuration.
        device: torch device.
        lora_adapter_path: if set, load saved LoRA weights from this path
            instead of initializing fresh LoRA. Used for checkpoint resume.
    """
    print(f"Loading actor model: {config.model_path}  (attn={config.attn_implementation})")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )

    # Must enable input gradients BEFORE LoRA wrapping
    model.enable_input_require_grads()

    # Gradient checkpointing trades compute for memory: recomputes forward during backward.
    # Disabling it halves backward time but requires ~2-3× more activation memory.
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if not config.use_lora:
        assert lora_adapter_path is None, (
            "lora_adapter_path passed but config.use_lora=False; these are incompatible. "
            "For FFT resume, `lora_adapter_path` should be None (full model weights come "
            "from model_path pointing at the checkpoint dir itself)."
        )
        print("FFT mode: training all parameters (no LoRA wrap)")
    elif lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        print(f"Loaded LoRA adapter from {lora_adapter_path}")
    else:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config)

    model = model.to(device)
    return model


def load_reference_model(
    config: TrainConfig,
    device: torch.device,
    lora_adapter_path: str | None = None,
) -> torch.nn.Module:
    """Load a frozen reference model for KL divergence computation.

    Args:
        lora_adapter_path: if set, apply this LoRA adapter to the base model
            and merge it in; the resulting merged model is the frozen reference.
            Used for RL-from-SFT so KL anchors the policy near the SFT init,
            not raw base.
    """
    print(f"Loading reference model: {config.model_path}  (attn={config.attn_implementation})")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )

    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
        print(f"Merged reference LoRA from {lora_adapter_path}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    return model


def load_tokenizer(config: TrainConfig) -> AutoTokenizer:
    """Load tokenizer with proper pad token configuration."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Optional: override chat_template + eos_token from a different model
    # (e.g. Qwen3-4B-Instruct template on a Qwen3-4B-Base model). Mirrors the
    # SFT-trainer flag. Needed for RL-from-SFT so the RL rollout uses the same
    # template the SFT model was trained on.
    if getattr(config, "chat_template_from", ""):
        src_tok = AutoTokenizer.from_pretrained(config.chat_template_from)
        assert src_tok.chat_template is not None, (
            f"chat_template_from={config.chat_template_from} has no chat_template"
        )
        tokenizer.chat_template = src_tok.chat_template
        if src_tok.eos_token_id is not None:
            tokenizer.eos_token = src_tok.eos_token
        print(f"Overriding chat_template from {config.chat_template_from}; "
              f"eos_token={tokenizer.eos_token!r} pad_token={tokenizer.pad_token!r}")
    return tokenizer
