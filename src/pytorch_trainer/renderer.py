"""Renderer abstraction for thinking model rollout.

Provides unified token-level rendering for Qwen3 (ChatML) and gpt-oss-20b (Harmony)
models with thinking/reasoning enabled. Each renderer knows how to:
1. Render the initial maze prompt to token IDs
2. Provide thinking prefill, stop tokens, and direction bridge tokens
3. Build turn continuation tokens (post-direction + env feedback + next prefix)

Token flow for one turn:
    full_token_ids += [thinking_prefill] + [thinking_output_with_stop] + [direction_bridge]
                      + [direction_token] + build_turn_continuation(env_text)
    response_position = len(full_token_ids_before) + len(thinking_prefill)
                        + len(thinking_output_with_stop) + len(direction_bridge)

For vLLM:
    Phase A prompt = full_token_ids + thinking_prefill
    Phase A generates thinking_output_with_stop (includes stop token)
    Phase B prompt = Phase A prompt + thinking_output_with_stop + direction_bridge
    Phase B generates 1 token (direction, action-masked)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .config import TrainConfig

logger = logging.getLogger(__name__)


def detect_model_format(tokenizer) -> str:
    """Detect model format from tokenizer vocabulary.

    Returns "harmony" for gpt-oss, "chatml" for Qwen3/ChatML models.
    """
    harmony_tokens = ["<|start|>", "<|channel|>", "<|return|>"]
    unk = getattr(tokenizer, "unk_token_id", None)
    has_harmony = all(
        tokenizer.convert_tokens_to_ids(t) != unk
        for t in harmony_tokens
    )
    if has_harmony:
        return "harmony"

    chatml_tokens = ["<|im_start|>", "<|im_end|>"]
    has_chatml = all(
        tokenizer.convert_tokens_to_ids(t) != unk
        for t in chatml_tokens
    )
    if has_chatml:
        return "chatml"

    raise ValueError(
        "Could not detect model format from tokenizer vocabulary. "
        "Expected Harmony (<|start|>, <|channel|>) or ChatML (<|im_start|>, <|im_end|>) tokens."
    )


class ThinkingRenderer(ABC):
    """Abstract base for thinking model renderers.

    Subclasses provide token-level primitives for the two-phase vLLM generation:
      Phase A: Generate thinking tokens (variable length, stop at end-of-thinking marker)
      Phase B: Generate direction token (1 token, action-masked to N/E/S/W)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def render_initial_prompt(self, messages: list[dict]) -> list[int]:
        """Render initial messages to token IDs with assistant generation prompt.

        The returned token sequence ends at the point where the assistant starts
        generating (before any thinking prefill).
        """

    @abstractmethod
    def get_thinking_prefill(self) -> list[int]:
        """Tokens to append to prompt before Phase A generation.

        Forces the model into the thinking channel:
        - ChatML: <think>\\n
        - Harmony: <|channel|>analysis<|message|>
        """

    @abstractmethod
    def get_thinking_stop_ids(self) -> list[int]:
        """Stop token IDs for Phase A. Generation stops when any of these appear.

        - ChatML: [</think>]
        - Harmony: [<|end|>]
        """

    @abstractmethod
    def get_response_stop_ids(self) -> list[int]:
        """Stop token IDs for single-phase generation (thinking + direction).

        The model generates the full response: thinking → bridge → direction → stop.
        vLLM stops at these tokens and strips them from output.
        The direction is the last token in the output.
        - ChatML: [<|im_end|>]
        - Harmony: [<|return|>]
        """

    @abstractmethod
    def get_direction_bridge(self) -> list[int]:
        """Tokens to insert between Phase A output and Phase B generation.

        These tokens "bridge" from the end of thinking to the position where
        the direction token should be generated.
        - ChatML: \\n (newline after </think>)
        - Harmony: <|start|>assistant<|channel|>final<|message|>
        """

    @abstractmethod
    def build_turn_continuation(self, env_text: str) -> list[int]:
        """Tokens after direction through end of next assistant prefix.

        Appended to full_token_ids after each non-final turn:
        - ChatML: <|im_end|>\\n<|im_start|>user\\n{env}<|im_end|>\\n<|im_start|>assistant\\n
        - Harmony: <|return|><|start|>user<|message|>{env}<|end|><|start|>assistant

        Does NOT include thinking_prefill — that's added separately at the
        start of the next turn's Phase A.
        """

    @abstractmethod
    def build_finalization(self) -> list[int]:
        """Tokens after the last direction (no continuation, just closing markers).

        - ChatML: <|im_end|>\\n
        - Harmony: <|return|>
        """

    def get_truncation_recovery_tokens(self) -> list[int] | None:
        """Tokens to force-append when thinking is truncated by max_thinking_tokens.

        When the model fills the thinking budget without producing a stop token,
        these tokens close the thinking block and bridge to the direction position,
        enabling a follow-up 1-token generation to recover the direction.

        Returns None if truncation recovery is not supported for this format.
        - ChatML: [</think>, \\n] — closes the think block, model generates direction next
        - Harmony: None — gpt-oss doesn't need this (low truncation rate)
        """
        return None  # default: no recovery


class ChatMLThinkingRenderer(ThinkingRenderer):
    """Qwen3 thinking renderer using ChatML format.

    Token structure per turn:
        <think>\\n{thinking}</think>\\n{direction}<|im_end|>\\n
        <|im_start|>user\\n{env}<|im_end|>\\n<|im_start|>assistant\\n
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Resolve special tokens
        self._think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
        assert len(self._think_open_ids) >= 1, f"<think>\\n encoded to empty: {self._think_open_ids}"

        self._think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        assert self._think_close_id is not None and self._think_close_id != getattr(
            tokenizer, "unk_token_id", None
        ), "Could not resolve </think> token ID"

        self._newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(self._newline_ids) == 1, f"\\n tokenizes to {len(self._newline_ids)} tokens"

        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert self._im_end_id is not None and self._im_end_id != getattr(
            tokenizer, "unk_token_id", None
        ), "Could not resolve <|im_end|> token ID"

        # Derive turn boundary tokens by diffing 1-turn vs 3-turn template
        msgs_1 = [{"role": "user", "content": "A"}]
        msgs_3 = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "A"},
        ]
        ids_1 = tokenizer.apply_chat_template(
            msgs_1, add_generation_prompt=True, tokenize=True,
        )
        ids_3 = tokenizer.apply_chat_template(
            msgs_3, add_generation_prompt=True, tokenize=True,
        )

        a_ids = tokenizer.encode("A", add_special_tokens=False)
        b_ids = tokenizer.encode("B", add_special_tokens=False)
        assert len(a_ids) == 1 and len(b_ids) == 1

        tail = ids_3[len(ids_1):]
        assert tail[: len(b_ids)] == b_ids
        remaining = tail[len(b_ids):]

        a_start = None
        for i in range(len(remaining) - len(a_ids) + 1):
            if remaining[i : i + len(a_ids)] == a_ids:
                a_start = i
                break
        assert a_start is not None, "Could not find marker A in template tail"

        # turn_prefix: <|im_end|>\n<|im_start|>user\n
        # turn_suffix: <|im_end|>\n<|im_start|>assistant\n
        self._turn_prefix = remaining[:a_start]
        self._turn_suffix = remaining[a_start + len(a_ids):]

        # Verify both start with im_end + newline
        im_end_nl = [self._im_end_id] + self._newline_ids
        assert self._turn_prefix[:2] == im_end_nl, (
            f"turn_prefix doesn't start with <|im_end|>\\n: {self._turn_prefix[:2]}"
        )
        assert self._turn_suffix[:2] == im_end_nl, (
            f"turn_suffix doesn't start with <|im_end|>\\n: {self._turn_suffix[:2]}"
        )

        logger.info(
            "ChatMLThinkingRenderer initialized: think_open=%s, think_close=%d, "
            "turn_prefix=%d toks, turn_suffix=%d toks",
            self._think_open_ids, self._think_close_id,
            len(self._turn_prefix), len(self._turn_suffix),
        )

    def render_initial_prompt(self, messages):
        return list(self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        ))

    def get_thinking_prefill(self):
        return list(self._think_open_ids)

    def get_thinking_stop_ids(self):
        return [self._think_close_id]

    def get_response_stop_ids(self):
        return [self._im_end_id]

    def get_direction_bridge(self):
        # After </think> (included in Phase A output), add \n before direction
        return list(self._newline_ids)

    def get_truncation_recovery_tokens(self):
        # </think>\n — closes the think block so the model can generate a direction
        return [self._think_close_id] + list(self._newline_ids)

    def build_turn_continuation(self, env_text):
        # After direction: <|im_end|>\n<|im_start|>user\n{env}<|im_end|>\n<|im_start|>assistant\n
        env_tokens = self.tokenizer.encode(env_text, add_special_tokens=False)
        return list(self._turn_prefix) + env_tokens + list(self._turn_suffix)

    def build_finalization(self):
        # Last turn: <|im_end|>\n
        return [self._im_end_id] + list(self._newline_ids)


class HarmonyThinkingRenderer(ThinkingRenderer):
    """gpt-oss-20b thinking renderer using Harmony format.

    Implements Harmony wire format directly (no tinker_cookbook dependency).

    Wire format: <|start|>role<|channel|>channel<|message|>content<|end|>
    System message: <|start|>system<|message|>{content}<|end|>
    User message:   <|start|>user<|message|>{content}<|end|>
    Assistant (gen): <|start|>assistant  (generation suffix, model continues)

    Token structure per turn:
        <|channel|>analysis<|message|>{thinking}<|end|>
        <|start|>assistant<|channel|>final<|message|>{direction}<|return|>
        <|start|>user<|message|>{env}<|end|><|start|>assistant
    """

    # System prompt template (matches GptOssRenderer.system_prompt_content)
    _SYSTEM_PROMPT_TEMPLATE = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        "Current date: {current_date}\n\n"
        "Reasoning: {reasoning_effort}\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message."
    )

    def __init__(self, tokenizer, config: TrainConfig):
        super().__init__(tokenizer)
        self._reasoning_effort = config.reasoning_effort

        # Resolve Harmony special tokens
        def _resolve(name: str) -> int:
            tid = tokenizer.convert_tokens_to_ids(name)
            assert tid is not None and tid != getattr(tokenizer, "unk_token_id", None), (
                f"Could not resolve {name} token ID"
            )
            return tid

        self._start_id = _resolve("<|start|>")
        self._channel_id = _resolve("<|channel|>")
        self._message_id = _resolve("<|message|>")
        self._end_id = _resolve("<|end|>")
        self._return_id = _resolve("<|return|>")

        # Pre-encode fixed token sequences
        self._analysis_prefill = tokenizer.encode(
            "<|channel|>analysis<|message|>", add_special_tokens=False,
        )
        self._direction_bridge = tokenizer.encode(
            "<|start|>assistant<|channel|>final<|message|>", add_special_tokens=False,
        )
        self._user_header = tokenizer.encode(
            "<|start|>user<|message|>", add_special_tokens=False,
        )
        self._assistant_header = tokenizer.encode(
            "<|start|>assistant", add_special_tokens=False,
        )
        self._system_header = tokenizer.encode(
            "<|start|>system<|message|>", add_special_tokens=False,
        )

        logger.info(
            "HarmonyThinkingRenderer initialized: effort=%s, analysis_prefill=%d toks, "
            "direction_bridge=%d toks",
            self._reasoning_effort, len(self._analysis_prefill), len(self._direction_bridge),
        )

    def render_initial_prompt(self, messages):
        """Render messages in Harmony wire format with system prompt.

        Produces: <|start|>system<|message|>{sysprompt}<|end|>
                  <|start|>user<|message|>{content}<|end|>
                  [... more messages ...]
                  <|start|>assistant
        """
        from datetime import datetime

        tokens = []

        # System prompt (always prepended for reasoning models)
        sys_content = self._SYSTEM_PROMPT_TEMPLATE.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            reasoning_effort=self._reasoning_effort,
        )
        tokens.extend(self._system_header)
        tokens.extend(self.tokenizer.encode(sys_content, add_special_tokens=False))
        tokens.append(self._end_id)

        # Render each message
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from content parts
                content = "".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )

            if role == "system":
                # User-provided system → developer role in Harmony
                header = self.tokenizer.encode(
                    "<|start|>developer<|message|>", add_special_tokens=False,
                )
                body = self.tokenizer.encode(
                    f"# Instructions\n\n{content}\n\n", add_special_tokens=False,
                )
            elif role == "user":
                header = list(self._user_header)
                body = self.tokenizer.encode(content, add_special_tokens=False)
            elif role == "assistant":
                # Historical assistant messages with extension property:
                # always end with <|return|> (not <|end|>)
                header = list(self._assistant_header)
                body = self.tokenizer.encode(
                    f"<|channel|>final<|message|>{content}<|return|>",
                    add_special_tokens=False,
                )
                tokens.extend(header)
                tokens.extend(body)
                continue  # skip the <|end|> append below
            else:
                header = self.tokenizer.encode(
                    f"<|start|>{role}<|message|>", add_special_tokens=False,
                )
                body = self.tokenizer.encode(content, add_special_tokens=False)

            tokens.extend(header)
            tokens.extend(body)
            tokens.append(self._end_id)

        # Generation suffix: <|start|>assistant
        tokens.extend(self._assistant_header)
        return tokens

    def get_thinking_prefill(self):
        # <|channel|>analysis<|message|>
        return list(self._analysis_prefill)

    def get_thinking_stop_ids(self):
        return [self._end_id]  # <|end|> = end of analysis channel

    def get_response_stop_ids(self):
        return [self._return_id]  # <|return|> = end of final channel

    def get_direction_bridge(self):
        # <|start|>assistant<|channel|>final<|message|>
        return list(self._direction_bridge)

    def build_turn_continuation(self, env_text):
        # After direction: <|return|><|start|>user<|message|>{env}<|end|><|start|>assistant
        env_tokens = self.tokenizer.encode(env_text, add_special_tokens=False)
        return (
            [self._return_id]
            + list(self._user_header)
            + env_tokens
            + [self._end_id]
            + list(self._assistant_header)
        )

    def build_finalization(self):
        return [self._return_id]  # <|return|>


def create_renderer(tokenizer, config: TrainConfig) -> ThinkingRenderer:
    """Create the appropriate thinking renderer based on model format.

    Auto-detects format from tokenizer vocabulary, or uses config.renderer_name
    as a hint for Harmony models.
    """
    fmt = detect_model_format(tokenizer)

    if fmt == "harmony":
        return HarmonyThinkingRenderer(tokenizer, config)
    elif fmt == "chatml":
        return ChatMLThinkingRenderer(tokenizer)
    else:
        raise ValueError(f"Unknown model format: {fmt}")
