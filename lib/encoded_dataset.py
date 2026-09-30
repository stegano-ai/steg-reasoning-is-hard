import functools
from dataclasses import dataclass
from typing import Callable

from lib.main_tasks.base import MainTask
from lib.side_tasks.base import SideTaskDefinition


@dataclass
class Sample:
    idx: int
    main_prompt: str                  # cover question for this item
    side_input: list                  # raw per-instance side-task input
    expected_reasoning: list[int]     # gold values to encode
    expected_answer: list[int] | None
    alphabet: list[int]               # values a reasoning position can take
    _render: Callable[[list[int]], str]
    main_answer: str | None = None    # gold natural answer to the cover question, for {MAIN_TASK_ANSWER}

    @property
    def max_value(self) -> int:
        return max(self.alphabet)

    def render(self, numbers: list[int]) -> str:
        """The cover text for this item with `numbers` encoded into it."""
        return self._render(numbers)


class EncodedDataset:
    """Pairs a main task with a side task into renderable samples. Tokenizer-free."""

    def __init__(self, main_task: MainTask, side_task: SideTaskDefinition, *, min_prefix: bool = False):
        if main_task.n_symbols is not None and main_task.n_symbols != len(side_task.alphabet):
            raise ValueError(
                f"main_task n_symbols={main_task.n_symbols} does not match "
                f"side task alphabet size {len(side_task.alphabet)}")
        self.main = main_task
        self.side = side_task
        self.min_prefix = min_prefix

    def train(self, limit: int | None = None) -> list[Sample]:
        return self._build(self.side.train_data, train=True, limit=limit)

    def test(self, limit: int | None = None) -> list[Sample]:
        return self._build(self.side.test_data, train=False, limit=limit)

    def _build(self, side_data: list[dict], *, train: bool, limit: int | None = None) -> list[Sample]:
        samples = []
        answers = self.main.answers(train=train)
        for i, question in enumerate(self.main.questions(train=train)):
            if limit is not None and len(samples) >= limit:
                break
            side = side_data[i % len(side_data)]
            values = side["reasoning"]
            # Filter on the full cover (cheap). The min-prefix search is deferred into _render
            # below, so only the few shots actually rendered pay for it — not the whole pool.
            trace = self.main.render(i, values, train=train, truncate_to=None)
            if trace is None:
                continue
            if self.main.encodes and self.main.decode(trace)[:len(values)] != values:
                continue
            trunc = values if self.min_prefix else None
            samples.append(Sample(
                idx=i,
                main_prompt=question,
                side_input=side["input"],
                expected_reasoning=values,
                expected_answer=side.get("output"),
                alphabet=self.side.alphabet,
                _render=functools.partial(self.main.render, i, train=train, truncate_to=trunc),
                main_answer=answers[i],
            ))
        return samples
