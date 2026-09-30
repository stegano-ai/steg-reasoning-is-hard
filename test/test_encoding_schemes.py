import pytest

from lib.encoding_schemes import (
    NoEncodingScheme,
    CapitalizationScheme,
    MarkdownScheme,
    NumberPrefixScheme,
    WhitespaceScheme,
    WordPrefixScheme,
    EvocativeWordPrefixScheme,
    SentenceMarkdownScheme,
    SentenceInterjectionScheme,
    SentenceNewlineScheme,
    make_general_scheme,
)


SAMPLE_TEXT = "The model provides a reasonable evaluation. However, it lacks depth in analyzing the specific rubric criteria. The response could be improved by addressing each point systematically."
SAMPLE_VALUES = [0, 1, 2, 0, 2, 1, 0, 1, 2, 0]

SENTENCE_SAMPLE = (
    "The model provides a reasonable evaluation. However, it lacks depth in the rubric criteria.\n"
    "The response could be improved. Each point should be addressed systematically. Overall the answer is partially correct."
)
SENTENCE_VALUES = [0, 1, 2, 0]


class TestCapitalization:
    scheme = CapitalizationScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded[:len(SAMPLE_VALUES)] == SAMPLE_VALUES

    def test_encoding_applies_correctly(self):
        text = "Hello world today"
        encoded = self.scheme.encode(text, [0, 1, 2])
        assert encoded == "Hello world TODAY"

    def test_skips_single_letter_words(self):
        text = "I am a good person"
        encoded = self.scheme.encode(text, [1, 2])
        decoded = self.scheme.decode(encoded)
        assert decoded[:2] == [1, 2]
        assert "am" in encoded.lower()
        assert "I" in encoded  # single letter untouched

    def test_more_values_than_words(self):
        text = "Hello world"
        encoded = self.scheme.encode(text, [0, 1, 2, 0, 1])
        decoded = self.scheme.decode(encoded)
        assert decoded[:2] == [0, 1]

    def test_fewer_values_than_words(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, [1, 2])
        decoded = self.scheme.decode(encoded)
        assert decoded[:2] == [1, 2]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestWhitespace:
    scheme = WhitespaceScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded[:len(SAMPLE_VALUES)] == SAMPLE_VALUES

    def test_encoding_applies_correctly(self):
        text = "one two three four"
        encoded = self.scheme.encode(text, [0, 2, 1])
        assert encoded == "one two   three  four"

    def test_newlines_become_spaces(self):
        text = "First line.\nSecond line."
        encoded = self.scheme.encode(text, [1, 0])
        decoded = self.scheme.decode(encoded)
        assert decoded[:2] == [1, 0]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)


class TestMarkdown:
    scheme = MarkdownScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded[:len(SAMPLE_VALUES)] == SAMPLE_VALUES

    def test_encoding_applies_correctly(self):
        text = "Hello world today"
        encoded = self.scheme.encode(text, [0, 1, 2])
        assert encoded == "Hello **world** _today_"

    def test_skips_single_letter_words(self):
        text = "I am a good person"
        encoded = self.scheme.encode(text, [1, 2])
        decoded = self.scheme.decode(encoded)
        assert decoded[:2] == [1, 2]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)


class TestWordPrefix:
    scheme = WordPrefixScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded == SAMPLE_VALUES

    def test_encoding_applies_correctly(self):
        text = "Hello world"
        encoded = self.scheme.encode(text, [1, 0, 2])
        assert encoded == "Fox Owl Penguin\nHello world"

    def test_decode_case_insensitive(self):
        decoded = self.scheme.decode("owl FOX Penguin\nbody")
        assert decoded == [0, 1, 2]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestEvocativeWordPrefix:
    scheme = EvocativeWordPrefixScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded == SAMPLE_VALUES

    def test_encoding_applies_correctly(self):
        text = "Hello world"
        encoded = self.scheme.encode(text, [1, 0, 2])
        assert encoded == "overshadow disclaim marinade\nHello world"

    def test_decode_case_insensitive(self):
        decoded = self.scheme.decode("disclaim OVERSHADOW Marinade\nbody")
        assert decoded == [0, 1, 2]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestFiveSymbolPrefixes:
    def test_roundtrip_values_0_to_4(self):
        for scheme in [NumberPrefixScheme(5), WordPrefixScheme(5), EvocativeWordPrefixScheme(5)]:
            encoded = scheme.encode(SAMPLE_TEXT, [3, 4, 0, 1, 2])
            assert scheme.decode(encoded) == [3, 4, 0, 1, 2]


class TestNSymbols:
    def test_markdown_five_symbol_roundtrip(self):
        for scheme in [MarkdownScheme(5), SentenceMarkdownScheme(5), WhitespaceScheme(5),
                       SentenceInterjectionScheme(5), SentenceNewlineScheme(5)]:
            encoded = scheme.encode(SENTENCE_SAMPLE, [3, 4, 1])
            assert scheme.decode(encoded)[:3] == [3, 4, 1]

    def test_markdown_bold_italic_and_code(self):
        encoded = MarkdownScheme(5).encode("Hello world today", [3, 4, 0])
        assert encoded == "***Hello*** `world` today"

    def test_description_only_mentions_used_symbols(self):
        assert "two" not in NumberPrefixScheme(2).description
        assert "italic" not in MarkdownScheme(2).description
        assert "'Next.'" not in SentenceInterjectionScheme(2).description

    def test_description_example_stays_in_alphabet(self):
        assert "[1, 0, 1]" in NumberPrefixScheme(2).description
        assert "[1, 0, 2]" in NumberPrefixScheme(5).description

    def test_encode_rejects_value_outside_alphabet(self):
        with pytest.raises(ValueError, match="alphabet is 0-1"):
            MarkdownScheme(2).encode("Hello world", [0, 2])

    def test_capitalization_rejects_n_above_3(self):
        with pytest.raises(ValueError, match="supports 2-3 symbols"):
            CapitalizationScheme(5)

class TestSentenceMarkdown:
    scheme = SentenceMarkdownScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SENTENCE_SAMPLE, SENTENCE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded[:len(SENTENCE_VALUES)] == SENTENCE_VALUES

    def test_encoding_applies_correctly(self):
        text = "The sum is twelve. Then we divide."
        encoded = self.scheme.encode(text, [1, 2])
        assert encoded == "**The sum is twelve**. _Then we divide_."

    def test_plain_text_decodes_to_zeros(self):
        assert self.scheme.decode("The sum is twelve. Then we divide.") == [0, 0]

    def test_more_values_than_sentences(self):
        encoded = self.scheme.encode("The sum is twelve.", [1, 2, 0])
        decoded = self.scheme.decode(encoded)
        assert decoded[:1] == [1]

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestSentenceInterjection:
    scheme = SentenceInterjectionScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SENTENCE_SAMPLE, SENTENCE_VALUES)
        assert self.scheme.decode(encoded) == SENTENCE_VALUES

    def test_encoding_applies_correctly(self):
        text = "The sum is twelve. Then we divide."
        encoded = self.scheme.encode(text, [2, 0])
        assert encoded == "The sum is twelve. Next. Then we divide. Okay."

    def test_lowercase_word_is_not_a_marker(self):
        assert self.scheme.decode("We look at what comes next. Then we stop.") == []

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestSentenceNewline:
    scheme = SentenceNewlineScheme(3)

    def test_roundtrip(self):
        encoded = self.scheme.encode(SENTENCE_SAMPLE, SENTENCE_VALUES)
        decoded = self.scheme.decode(encoded)
        assert decoded[:len(SENTENCE_VALUES)] == SENTENCE_VALUES

    def test_encoding_applies_correctly(self):
        text = "The sum is twelve. Then we divide."
        encoded = self.scheme.encode(text, [1, 0])
        assert encoded == "The sum is twelve.\n\nThen we divide.\n"

    def test_description_is_string(self):
        assert isinstance(self.scheme.description, str)
        assert len(self.scheme.description) > 20


class TestMakeGeneralScheme:
    def test_all_schemes_accessible(self):
        for name in ["capitalization", "whitespace", "markdown", "number_prefix", "word_prefix", "evocative_word_prefix",
                     "sentence_markdown", "sentence_interjection", "sentence_newline"]:
            scheme = make_general_scheme(name, 3)
            assert hasattr(scheme, "encode")
            assert hasattr(scheme, "decode")
            assert hasattr(scheme, "description")

    def test_all_schemes_roundtrip(self):
        for name in ["capitalization", "whitespace", "markdown", "number_prefix", "word_prefix", "evocative_word_prefix"]:
            scheme = make_general_scheme(name, 3)
            encoded = scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES)
            decoded = scheme.decode(encoded)
            assert decoded[:len(SAMPLE_VALUES)] == SAMPLE_VALUES, f"Scheme {name} failed roundtrip"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding scheme type"):
            make_general_scheme("nonexistent", 3)

    def test_none_and_none_string_give_no_encoding(self):
        for name in (None, "none"):
            scheme = make_general_scheme(name, 3)
            assert isinstance(scheme, NoEncodingScheme)
            assert scheme.encode(SAMPLE_TEXT, SAMPLE_VALUES) == SAMPLE_TEXT
            assert scheme.decode(SAMPLE_TEXT) == []
