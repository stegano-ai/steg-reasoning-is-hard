from ._standard import StandardSchemeTask


class BlankTask(StandardSchemeTask):
    """Empty cover: no question, no cover text. For side-task-only baselines where the
    model just answers the side task (e.g. a #### line) with nothing to blend into."""

    name = "blank"

    def __init__(self, scheme: str, n_train: int = 500, n_test: int = 500, n_symbols: int = 3):
        super().__init__(scheme, n_symbols)
        self._n_train = n_train
        self._n_test = n_test

    def _natural_pool(self, *, train: bool) -> list[tuple[str, str]]:
        return [("", "")] * (self._n_train if train else self._n_test)

    def check_answer(self, prompt: str, model_output: str, label: str) -> bool:
        return True
