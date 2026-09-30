import asyncio
import os
import weakref
from typing import Protocol

# One cap on concurrent OpenRouter requests, shared by every sampler and the CoT monitor
# within an event loop. Harmless below the cap (a single cell is ~n_eval requests); it
# bounds the burst when many cells run in one loop (a full sweep — see
# experiments/prompting/main.py). Kept per-loop so callers that drive the monitor across
# several asyncio.run() calls (e.g. RL training steps) each get a valid semaphore instead
# of reusing one bound to a stale loop.
MAX_OPENROUTER_CONCURRENCY = 40
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def openrouter_semaphore() -> asyncio.Semaphore:
    """The shared OpenRouter limiter for the currently running event loop."""
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(MAX_OPENROUTER_CONCURRENCY)
        _semaphores[loop] = sem
    return sem


class ContentFilterError(RuntimeError):
    """The provider's content filter truncated the response; the sample is unusable."""


def parse_response_text(renderer, tokens) -> str:
    """Parse sampled tokens via a tinker_cookbook renderer into plain text.

    Renderers (e.g. Qwen3.5) return `content` as a list of ThinkingPart/TextPart
    dicts instead of a str whenever the completion contains a `<think>` or
    `<tool_call>` block. Callers that treat completions as plain text (scoring,
    logging) need the flattened string form.
    """
    msg, _ = renderer.parse_response(tokens)
    content = msg["content"]
    if isinstance(content, list):
        content = "".join(
            f"<think>{p['thinking']}</think>" if p.get("type") == "thinking" else p.get("text", "")
            for p in content if isinstance(p, dict)
        )
    return content


class Sampler(Protocol):
    async def sample(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        prefill: str | None = None,
    ) -> str: ...


class OpenRouterSampler:
    def __init__(self, model: str, reasoning: dict | None = None,
                 provider: str | None = None):
        from openai import AsyncOpenAI
        self.model = model
        # default disables reasoning; models that mandate it (e.g. Gemini) override with
        # a minimal-effort setting that still produces no reasoning tokens
        self.reasoning = reasoning if reasoning is not None else {"enabled": False}
        # pin to one provider so results aren't a mix of quantizations/serving stacks
        self.extra_body = {"reasoning": self.reasoning}
        if provider:
            self.extra_body["provider"] = {"only": [provider], "allow_fallbacks": False}
        self._client = AsyncOpenAI(
            api_key=os.environ["OPEN_ROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=300,  # cot cells generate up to 16k reasoning tokens; 90s times out mid-generation
            max_retries=8,  # ride out rate-limited providers (e.g. Crusoe shared-pool 429s) with backoff
        )

    async def sample(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        prefill: str | None = None,
    ) -> str:
        if prefill:
            messages = messages + [{"role": "assistant", "content": prefill}]
        async with openrouter_semaphore():
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                extra_body=self.extra_body,
            )
        choices = response.choices or []
        if choices and choices[0].finish_reason == "content_filter":
            # provider safety filter truncated the response (common for edgy WildChat covers
            # on Anthropic models). Not a task failure — raise so the eval drops this sample.
            raise ContentFilterError(f"content_filter ({self.model})")
        completion = (choices[0].message.content if choices else "") or ""
        if not completion:
            # transient provider glitch, or reasoning ate the whole max_tokens budget
            # (finish_reason=length with empty content). Raise so the eval drops the sample
            # instead of silently scoring an empty response.
            finish = choices[0].finish_reason if choices else "no choices"
            raise RuntimeError(f"empty completion ({self.model}, finish_reason={finish})")
        return completion


class TinkerSampler:
    def __init__(self, sampling_client, renderer):
        self.sampling_client = sampling_client
        self.renderer = renderer

    async def sample(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        prefill: str | None = None,
    ) -> str:
        import tinker
        prompt = self.renderer.build_generation_prompt(
            messages, "assistant", prefill)
        renderer_stops = [self.renderer.tokenizer.decode([tid]) for tid in self.renderer.get_stop_sequences()]
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer_stops + (stop or []),
        )
        resp = await self.sampling_client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=params,
        )
        return parse_response_text(self.renderer, resp.sequences[0].tokens)

    @classmethod
    async def from_checkpoint(
        cls,
        *,
        base_model: str,
        model_path: str | None = None,
        renderer_name: str | None = None,
    ) -> "TinkerSampler":
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer_name = renderer_name or model_info.get_recommended_renderer_name(base_model)
        tokenizer = get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        service_client = tinker.ServiceClient()
        sampling_client = await service_client.create_sampling_client_async(
            **({"model_path": model_path} if model_path else {"base_model": base_model}),
        )
        return cls(sampling_client=sampling_client, renderer=renderer)


async def make_sampler(cfg: dict) -> Sampler:
    kind = cfg["kind"]
    if kind == "openrouter":
        return OpenRouterSampler(model=cfg["model"], reasoning=cfg.get("reasoning"),
                                 provider=cfg.get("provider"))
    if kind == "tinker":
        return await TinkerSampler.from_checkpoint(
            base_model=cfg["base_model"],
            model_path=cfg.get("model_path"),
            renderer_name=cfg.get("renderer_name"),
        )
    raise ValueError(f"Unknown sampler kind: {kind}")
