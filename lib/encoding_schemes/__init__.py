from .schemes import (
    SteganographyScheme,
    NoEncodingScheme,
    CapitalizationScheme,
    WhitespaceScheme,
    MarkdownScheme,
    NumberPrefixScheme,
    WordPrefixScheme,
    EvocativeWordPrefixScheme,
    SentenceMarkdownScheme,
    SentenceInterjectionScheme,
    SentenceNewlineScheme,
)

_STANDARD = {
    "capitalization": CapitalizationScheme,
    "whitespace": WhitespaceScheme,
    "markdown": MarkdownScheme,
    "number_prefix": NumberPrefixScheme,
    "word_prefix": WordPrefixScheme,
    "evocative_word_prefix": EvocativeWordPrefixScheme,
    "sentence_markdown": SentenceMarkdownScheme,
    "sentence_interjection": SentenceInterjectionScheme,
    "sentence_newline": SentenceNewlineScheme,
}


def make_general_scheme(name: str | None, n_symbols: int) -> SteganographyScheme:
    """Build a standard scheme by name.
    `None` or "none" yields NoEncodingScheme (the response carries no hidden values)."""
    if name is None or name == "none":
        return NoEncodingScheme(n_symbols)
    if name not in _STANDARD:
        raise ValueError(f"Unknown encoding scheme type: {name}")
    return _STANDARD[name](n_symbols)


def min_roundtrip_prefix(text: str, scheme: SteganographyScheme, values: list[int]) -> str | None:
    """Smallest word-prefix of `text` that still round-trips `values` through `scheme`.

    The round-trip is monotonic in prefix length — once a prefix carries all n values, no
    longer prefix drops them — so we gallop to a working upper bound then binary-search for
    the smallest working length. That is O(log |words|) encode/decodes instead of O(|words|),
    which matters for the slow (sentence-level) schemes.
    """
    words = text.split()
    n = len(values)
    total = len(words)
    if total < n + 1:  # too short to form a prefix of the required length
        return None

    def ok(k: int) -> bool:
        return scheme.decode(scheme.encode(" ".join(words[:k]), values))[:n] == values

    lo, hi = n, n + 1  # lo = known-too-short, hi = current candidate (>= n+1)
    while hi < total and not ok(hi):
        lo, hi = hi, min(hi * 2, total)
    if not ok(hi):
        return None
    while lo + 1 < hi:  # smallest working length in (lo, hi]
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return " ".join(words[:hi])
