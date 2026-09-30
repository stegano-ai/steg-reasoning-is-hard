from abc import abstractmethod

from lib.encoding_schemes import make_general_scheme, min_roundtrip_prefix

from .base import MainTask


class StandardSchemeTask(MainTask):
    """Base for tasks whose only schemes are general (post-encode a natural answer)."""

    def __init__(self, scheme: str, n_symbols: int = 3):
        super().__init__(scheme)
        self._scheme = make_general_scheme(scheme, n_symbols)
        self.encodes = self._scheme.encodes
        self.n_symbols = n_symbols if self.encodes else None

    @abstractmethod
    def _natural_pool(self, *, train: bool) -> list[tuple[str, str]]:
        ...

    def _pool(self, *, train: bool = True) -> list[tuple[str, str]]:
        return self._natural_pool(train=train)

    def render(self, idx: int, encoded_numbers: list[int], *, train: bool = True,
               truncate_to: list[int] | None = None) -> str | None:
        natural = self._natural_pool(train=train)[idx][1]
        if truncate_to is not None:
            natural = min_roundtrip_prefix(natural, self._scheme, truncate_to)
            if natural is None:
                return None
        return self._scheme.encode(natural, encoded_numbers)

    def decode(self, text: str) -> list[int]:
        return self._scheme.decode(text)

    @property
    def description(self) -> str:
        return self._scheme.description
