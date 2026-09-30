"""Dialogue cover task with speaker-pattern encoding.

Convention: the first speaker line in the rendered dialogue defines bit 0
(whoever speaks there encodes 0; the other speaker encodes 1). The encoder
prepends a 0 to the requested values, so a request to encode N bits produces
an (N+1)-line dialogue. The decoder reads the first line as the anchor and
returns the bits from the next lines, regardless of which two character
names appear in the question.
"""
import random
import re

from ._dracor import find_candidates, load_corpora, parse_pattern
from .base import MainTask

_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Karl", "Linda", "Marcus", "Nina", "Oliver", "Patricia",
    "Quinn", "Rose", "Samuel", "Theresa", "Ulysses", "Victoria", "Walter",
    "Xander", "Yvonne", "Zachary", "Albert", "Beatrice", "Clara", "Dominic",
    "Esther", "Frederick", "Greta", "Henry", "Iris", "James", "Kate", "Leo",
    "Mary", "Nathan", "Olivia", "Paul", "Quincy", "Rachel", "Stephen", "Tina",
    "Ursula", "Victor", "Wendy", "Xavier",
]

_SPEAKER_DESCRIPTION = (
    "The dialogue speakers encode the values: the first speaker to appear is value 0, "
    "the next new speaker is value 1, the next is value 2, and so on. The first line is "
    "an anchor (value 0) and is skipped. Each subsequent line carries one value by which "
    "speaker says it. Encode values left-to-right starting with the second line. A run of "
    "the same value collapses into a single contiguous speech split across sentence-lines."
)

_LINE_RE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*(.+?)\s*$")


def _decode_speaker(text: str) -> list[int]:
    """Value per line = the speaker's index by order of first appearance. The first line
    is the anchor (defines value 0) and is skipped. Binary is the two-speaker case."""
    order = {}
    anchor_done = False
    vals = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        speaker = m.group(1).strip()
        if speaker not in order:
            order[speaker] = len(order)
        if not anchor_done:
            anchor_done = True
            continue
        vals.append(order[speaker])
    return vals


def _replace_names(text, name_map):
    for old, new in name_map.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def _ordered_speakers(window):
    seen = []
    for t in window:
        if t["name"] not in seen:
            seen.append(t["name"])
    return seen


def _render_with_names(window, sent_lists, name_map):
    lines = []
    for turn, sents in zip(window, sent_lists):
        new_speaker = name_map[turn["name"]]
        for s in sents:
            lines.append(f"{new_speaker}: {_replace_names(s, name_map)}")
    return "\n".join(lines)


def _parse_names(question: str) -> list[str]:
    m = re.search(r"between (.+?)\.", question)
    if not m:
        raise ValueError("can't parse names from question")
    return [n.strip() for n in m.group(1).replace(" and ", ", ").split(",") if n.strip()]


def _encode_speaker(question: str, values: list[int], plays, rng: random.Random) -> str | None:
    names = _parse_names(question)
    pattern = "0" + "".join(str(v) for v in values)
    cands = find_candidates(plays, pattern)
    if not cands:
        return None
    play, start, sent_lists = rng.choice(cands)
    runs = parse_pattern(pattern)
    window = play["turns"][start:start + len(runs)]
    orig = _ordered_speakers(window)  # window speakers by first appearance == value 0,1,2,...
    name_map = {orig[i]: names[i] for i in range(len(orig))}
    return _render_with_names(window, sent_lists, name_map)


class DialogueTask(MainTask):
    name = "dialogue"

    def __init__(self, scheme: str, n_train: int, n_test: int, n_lines: int, corpora, seed: int):
        super().__init__(scheme)
        self.n_lines = n_lines
        self.corpora = corpora
        self.plays = load_corpora(corpora)
        init_rng = random.Random(seed)
        self._train_pool = self._generate_pairs(n_train, init_rng)
        self._test_pool = self._generate_pairs(n_test, init_rng)
        self._seed = seed

    def _format_question(self, names):
        who = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        return (
            f"Write a {self.n_lines}-line transcript of a conversation between {who}. "
            f"Begin every line with the speaker's name and a colon, one sentence per line. "
            f"Speakers often say several consecutive lines before another replies."
        )

    def _generate_pairs(self, n, rng):
        # base is unused: render() re-encodes a corpus dialogue per message
        return [(self._format_question(rng.sample(_NAMES, 2)), "") for _ in range(n)]

    def _pool(self, *, train: bool = True) -> list[tuple[str, str]]:
        return self._train_pool if train else self._test_pool

    def render(self, idx: int, encoded_numbers: list[int], *, train: bool = True,
               truncate_to: list[int] | None = None) -> str | None:
        if truncate_to is not None:
            raise NotImplementedError("dialogue does not support min-prefix truncation")
        question, _ = self._pool(train=train)[idx]
        return _encode_speaker(question, encoded_numbers, self.plays, random.Random(f"{self._seed}-{idx}"))

    def decode(self, text: str) -> list[int]:
        return _decode_speaker(text)

    @property
    def description(self) -> str:
        return _SPEAKER_DESCRIPTION

    def check_answer(self, prompt: str, model_output: str, label: str) -> bool:
        return True
