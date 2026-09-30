import re

from datasets import load_dataset

from ._standard import StandardSchemeTask


def _format_question(question: str, options: list[str]) -> str:
    opts = "\n".join(options)
    return f"{question}\n\n{opts}"


def _format_answer(rationale: str, correct: str) -> str:
    lines = rationale.strip().split("\n")[:-1]
    cleaned = "\n".join(lines).strip()
    return f"{cleaned}\n#### {correct}"


def _extract_final(text: str) -> str:
    match = re.search(r"####\s*(.+)", text)
    return match.group(1).strip().upper() if match else ""


def _load_split(split: str) -> list[tuple[str, str]]:
    ds = load_dataset("deepmind/aqua_rat", split=split)
    return [
        (_format_question(ex["question"], ex["options"]), _format_answer(ex["rationale"], ex["correct"]))
        for ex in ds
    ]


class AquaRatTask(StandardSchemeTask):
    name = "aqua_rat"

    def __init__(self, scheme: str, n_symbols: int = 3):
        super().__init__(scheme, n_symbols)
        self._train = _load_split("train")
        self._test = _load_split("test")

    def _natural_pool(self, *, train: bool) -> list[tuple[str, str]]:
        return self._train if train else self._test

    def check_answer(self, prompt: str, model_output: str, label: str) -> bool:
        return _extract_final(model_output) == _extract_final(label)
