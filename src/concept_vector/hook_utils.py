"""PyTorch hook utilities for activation extraction."""

import torch
import contextlib
import functools

from typing import List, Optional, Tuple, Callable


# ── Steer-mask utilities (assistant-turn-only steering) ────────────────────


def get_user_turn_marker_ids(tokenizer, format_str: str) -> Optional[List[int]]:
    """Return token IDs that open a user turn for the given chat format.

    Returns None if the format has no role-marker scaffolding ("plain").
    """
    if format_str == "chatml":
        marker = "<|im_start|>user"
    elif format_str == "harmony":
        marker = "<|start|>user"
    else:
        return None
    ids = tokenizer.encode(marker, add_special_tokens=False)
    assert len(ids) > 0, f"Empty tokenization for marker {marker!r}"
    return ids


def get_turn_end_token_id(tokenizer, format_str: str) -> Optional[int]:
    """Return the token ID that closes a turn for the given chat format."""
    if format_str == "chatml":
        tok = "<|im_end|>"
    elif format_str == "harmony":
        tok = "<|end|>"
    else:
        return None
    tid = tokenizer.convert_tokens_to_ids(tok)
    unk = tokenizer.unk_token_id
    assert tid is not None and tid != unk, (
        f"Tokenizer is missing the {tok!r} special token; cannot build "
        f"steer_mask for format={format_str!r}"
    )
    return tid


def compute_user_turn_steer_mask(
    input_ids: torch.Tensor,
    user_marker_ids: List[int],
    end_token_id: int,
) -> torch.Tensor:
    """Return a 1D bool mask aligned with `input_ids`.

    True  = steer this position (assistant / system / scaffolding).
    False = do NOT steer (token is inside a user turn).

    A user turn spans from the first token of `user_marker_ids` through the
    next `end_token_id`, inclusive. The trailing newline / role-start of the
    next turn is treated as scaffolding and remains steered.
    """
    assert input_ids.dim() == 1, f"Expected 1D input_ids, got {input_ids.shape}"
    n = input_ids.shape[0]
    m = len(user_marker_ids)
    mask = torch.ones(n, dtype=torch.bool)
    if m == 0 or n == 0:
        return mask

    ids_list = input_ids.tolist()
    marker = list(user_marker_ids)

    i = 0
    while i + m <= n:
        if ids_list[i:i + m] == marker:
            j = i + m
            while j < n and ids_list[j] != end_token_id:
                j += 1
            if j < n:
                mask[i:j + 1] = False
                i = j + 1
            else:
                # No closing end-of-turn token found — defensively treat the
                # rest of the sequence as user content. Should not happen for
                # well-formed chat-templated prompts.
                mask[i:] = False
                break
        else:
            i += 1

    return mask


def build_batched_steer_mask(
    batched_input_ids: torch.Tensor,
    batch_input_lens: List[int],
    tokenizer,
    format_str: str,
) -> Optional[torch.Tensor]:
    """Build a (B, L) bool steer-mask for a left-padded batch.

    Pad positions are False (won't be steered, harmless since attention
    masks them anyway). Inside each sample's actual content, positions are
    True except inside user turns.

    Returns None for formats without role markers (e.g., "plain").
    """
    marker_ids = get_user_turn_marker_ids(tokenizer, format_str)
    if marker_ids is None:
        return None
    end_id = get_turn_end_token_id(tokenizer, format_str)

    B, L = batched_input_ids.shape
    assert len(batch_input_lens) == B, (
        f"batch_input_lens has {len(batch_input_lens)} entries but batch is {B}"
    )

    mask = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        actual_len = batch_input_lens[b]
        assert actual_len <= L, f"Sample {b}: actual_len {actual_len} > L {L}"
        pad_len = L - actual_len
        unpadded = batched_input_ids[b, pad_len:].cpu()
        per_sample = compute_user_turn_steer_mask(unpadded, marker_ids, end_id)
        mask[b, pad_len:] = per_sample
    return mask


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
        forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
        forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()
