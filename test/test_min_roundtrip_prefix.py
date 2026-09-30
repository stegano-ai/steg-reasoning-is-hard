import pytest

from lib.encoding_schemes import (
    CapitalizationScheme,
    MarkdownScheme,
    NumberPrefixScheme,
    WhitespaceScheme,
    WordPrefixScheme,
    min_roundtrip_prefix,
)


SAMPLE_TEXT = (
    "The model provides a reasonable evaluation. However, it lacks depth in analyzing "
    "the specific rubric criteria. The response could be improved by addressing each "
    "point systematically."
)


@pytest.mark.parametrize(
    "scheme",
    [CapitalizationScheme(3), WhitespaceScheme(3), MarkdownScheme(3), WordPrefixScheme(3), NumberPrefixScheme(3)],
)
class TestMinRoundtripPrefix:
    def test_prefix_is_word_prefix_of_text(self, scheme):
        values = [0, 1, 2, 0, 1]
        prefix = min_roundtrip_prefix(SAMPLE_TEXT, scheme, values)
        assert prefix is not None
        assert SAMPLE_TEXT.startswith(prefix)
        assert prefix == " ".join(prefix.split())  # no internal whitespace mangling

    def test_prefix_roundtrips(self, scheme):
        values = [0, 1, 2, 0, 1]
        prefix = min_roundtrip_prefix(SAMPLE_TEXT, scheme, values)
        assert prefix is not None
        assert scheme.decode(scheme.encode(prefix, values))[:len(values)] == values

    def test_no_shorter_prefix_works(self, scheme):
        values = [0, 1, 2, 0, 1]
        prefix = min_roundtrip_prefix(SAMPLE_TEXT, scheme, values)
        assert prefix is not None
        words = prefix.split()
        if len(words) <= len(values) + 1:
            return
        shorter = " ".join(words[:-1])
        assert scheme.decode(scheme.encode(shorter, values))[:len(values)] != values

    def test_returns_none_when_text_too_short(self, scheme):
        values = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        prefix = min_roundtrip_prefix("one two", scheme, values)
        assert prefix is None


def test_empty_text_returns_none():
    assert min_roundtrip_prefix("", WhitespaceScheme(3), [0, 1]) is None


def test_single_value():
    prefix = min_roundtrip_prefix(SAMPLE_TEXT, WhitespaceScheme(3), [1])
    assert prefix is not None
    assert WhitespaceScheme(3).decode(WhitespaceScheme(3).encode(prefix, [1]))[:1] == [1]
