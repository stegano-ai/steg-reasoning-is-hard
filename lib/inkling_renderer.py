"""Thinking-disabled renderer for Inkling, registered as `tml_v0_no_thinking`.

Every other model family in the sweep ships a `*_disable_thinking` renderer. Inkling does
not: `tml_v0` takes a continuous `effort` in [0, 1) that defaults to 0.9, and the generic
builders never pass it — `conversation_to_datum` calls `build_supervised_example` with no
effort, so training silently renders at 0.9. This subclass pins effort to 0.

Why that matters here: score_local decodes the encoded values out of the assistant's
`content`. At effort 0 the model turn is `<|message_model|><|content_text|>...`, and
parse_response returns that text as `content`. With thinking on, the answer lands in a
reasoning segment scoring can't see.

Imported for its registration side effect by the sft training functions.
"""

from __future__ import annotations

import tinker
import torch

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import Message, Role, TrainOnWhat
from tinker_cookbook.renderers.tml_v0 import TmlRenderInput, TmlV0Renderer

RENDERER_NAME = "tml_v0_no_thinking"
NO_THINKING_EFFORT = 0.0


class TmlV0NoThinkingRenderer(TmlV0Renderer):
    def build_generation_prompt(
        self,
        messages: list[Message] | TmlRenderInput,
        role: Role = "assistant",
        prefill: str | None = None,
        effort: float = NO_THINKING_EFFORT,
    ) -> tinker.ModelInput:
        if prefill is None:
            return super().build_generation_prompt(messages, role, prefill, effort)
        # tml_v0 refuses prefill (its sampling API can't continue a partial message), but
        # prefill_datum only needs the TOKENS of an unclosed assistant turn: the supervised
        # rendering truncated right after the prefill text, before the closing specials.
        full, _ = self.build_supervised_example(
            [*messages, {"role": "assistant", "content": prefill}],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=effort)
        tokens = full.to_ints()
        text = self.tokenizer.encode(prefill, add_special_tokens=False)
        start = next(i for i in range(len(tokens) - len(text), -1, -1)
                     if tokens[i:i + len(text)] == text)
        return tinker.ModelInput.from_ints(tokens[:start + len(text)])

    def build_supervised_examples(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = NO_THINKING_EFFORT,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        return super().build_supervised_examples(messages, train_on_what, effort)

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = NO_THINKING_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(messages, train_on_what, effort)


renderers.register_renderer(
    RENDERER_NAME, lambda tokenizer, image_processor: TmlV0NoThinkingRenderer(tokenizer)
)
