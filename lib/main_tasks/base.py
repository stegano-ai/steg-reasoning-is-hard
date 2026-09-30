from abc import ABC, abstractmethod


class MainTask(ABC):
    """A cover task: turns base item ``idx`` into a reasoning trace with values encoded
    into it. Knows nothing about side tasks — pairing lives in lib.encoded_dataset."""

    name: str
    encodes: bool = True
    n_symbols: int | None = None  # alphabet size of a general encoding scheme, if the task uses one

    def __init__(self, scheme: str):
        self._scheme_name = scheme

    @abstractmethod
    def _pool(self, *, train: bool = True) -> list[tuple[str, str]]:
        """The (question, base_cover) pairs this task draws reasoning traces from."""
        ...

    def questions(self, *, train: bool = True) -> list[str]:
        return [q for q, _ in self._pool(train=train)]

    def answers(self, *, train: bool = True) -> list[str]:
        return [a for _, a in self._pool(train=train)]

    @abstractmethod
    def render(self, idx: int, encoded_numbers: list[int], *, train: bool = True,
               truncate_to: list[int] | None = None) -> str | None:
        """The reasoning trace for base item ``idx`` with ``encoded_numbers`` encoded into it.

        The base (cover text / random structure) depends only on ``idx``, so calling
        ``render`` twice for the same ``idx`` with different ``encoded_numbers`` yields
        traces that differ only where the changed values are encoded. Returns ``None``
        when the base cannot carry the requested values.

        ``truncate_to`` (SFT min-prefix): if given, the cover is first shortened to the
        smallest prefix that still round-trips ``truncate_to`` values, then ``encoded_numbers``
        are encoded into that fixed prefix. Pass the gold values as ``truncate_to`` so the
        normal / shuffled / unencoded traces all share one truncated cover.
        """
        ...

    @abstractmethod
    def decode(self, text: str) -> list[int]:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def check_answer(self, prompt: str, model_output: str, label: str) -> bool:
        ...
