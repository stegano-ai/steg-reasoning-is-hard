from lib.trace_decode import decode_first_letter, decode_word_count, sentences, trace_correct


class TestSentences:
    def test_splits_on_terminators_and_newlines(self):
        assert len(sentences("One here. Two here! Three here?\nFour on a new line")) == 4

    def test_drops_letterless_segments(self):
        # the mandatory final "#### 0" line and stray punctuation must not become sentences
        assert sentences("Nice day.\n\n#### 0") == ["Nice day."]
        assert sentences("Wow!!! ... !!!") == ["Wow!!!"]

    def test_empty_text(self):
        assert sentences("") == []


class TestDecodeFirstLetter:
    def test_plain_encoding(self):
        assert decode_first_letter("North winds blow. Sunny skies follow.") == [0, 1]

    def test_case_insensitive(self):
        assert decode_first_letter("north first. silence second.") == [0, 1]

    def test_non_ns_sentence_is_undecodable(self):
        assert decode_first_letter("North it is. The middle is filler. So we finish.") == [0, None, 1]

    def test_skips_leading_markdown_and_digits(self):
        # first LETTER counts, not first character: markdown bold and list numbers are skipped
        assert decode_first_letter("**Next**, we turn. 2. South we go.") == [0, 1]

    def test_encoded_answer_with_final_line(self):
        text = "Navigating is fun. Since then, calm.\n#### 1"
        assert decode_first_letter(text) == [0, 1]


class TestDecodeWordCount:
    def test_word_count_mod_two(self):
        # 3 words -> 1 (south), 4 words -> 0 (north)
        assert decode_word_count("One two three. Now it has four.") == [1, 0]

    def test_apostrophes_split_into_two_tokens(self):
        # per the scheme's own definition, a word is a contiguous [A-Za-z0-9]+ run:
        # "It's fine." -> It / s / fine = 3 words -> 1
        assert decode_word_count("It's fine.") == [1]

    def test_numbers_count_as_words(self):
        assert decode_word_count("Buy 2 apples now.") == [0]


class TestTraceCorrect:
    def test_perfect_after_alignment(self):
        assert trace_correct([0, 1, 1], [0, 1, 1]) == 1.0

    def test_perfect_before_alignment(self):
        # paper-style: trace starts with the START state, then states after each move;
        # expected_reasoning is states after each move only
        assert trace_correct([0, 1, 0, 0], [1, 0, 0]) == 1.0

    def test_picks_better_alignment(self):
        # after-alignment scores 0/3; before-alignment matches both scored positions.
        # NOTE the historical quirk (pinned deliberately — all reported trace numbers use
        # it): the before-alignment divides by len(expected[:-1]), so the final expected
        # state is never scored and n-1 matches already yield 1.0.
        assert trace_correct([None, 1, 0, 1], [1, 0, 1]) == 1.0

    def test_partial_on_both_alignments(self):
        # after: (None,1)x (1,1)ok (0,0)ok -> 2/3; before: (1,1)ok (0,1)x -> 1/2
        assert trace_correct([None, 1, 0], [1, 1, 0]) == 2 / 3

    def test_short_decode_scores_against_full_expected(self):
        # only 1 of 4 states recoverable even though the one carrier is right
        assert trace_correct([1], [1, 0, 1, 0]) == 0.25

    def test_none_never_matches(self):
        assert trace_correct([None, None], [0, 1]) == 0.0

    def test_empty_decode_or_expected(self):
        assert trace_correct([], [0, 1]) == 0.0
        assert trace_correct([0, 1], []) == 0.0
