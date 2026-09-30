from lib.encoded_dataset import EncodedDataset
from lib.main_tasks.knapsack import KnapsackTask
from lib.prefill import prefill_committing_value
from lib.side_tasks import get_side_task


class CharTok:
    """Char-level stand-in tokenizer: one token per character, deterministic, offline."""
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids): return "".join(chr(i) for i in ids)


def _fixture():
    main = KnapsackTask(scheme="binary_2", n_train=8, n_test=4, n_items=5,
                        capacity=15, max_value=99, max_weight=10, n_candidates=4, seed=1)
    side = get_side_task(type="rolling_mod_sum", n_samples=50, seq_length=4,
                         operand_range=5, modulus=4)
    return main, EncodedDataset(main, side).train()[0]


class TestPrefillCommittingValue:
    def test_prefill_is_a_prefix_of_the_full_trace(self):
        (main, s), tok = _fixture(), CharTok()
        wrong = (s.expected_reasoning[1] + 1) % 4
        pre = prefill_committing_value(s, 1, wrong, tok, main.decode)
        numbers = list(s.expected_reasoning); numbers[1] = wrong
        assert s.render(numbers).startswith(pre)

    def test_wrong_and_correct_prefill_differ(self):
        (main, s), tok = _fixture(), CharTok()
        correct = s.expected_reasoning[1]
        wrong = (correct + 1) % 4
        assert (prefill_committing_value(s, 1, correct, tok, main.decode)
                != prefill_committing_value(s, 1, wrong, tok, main.decode))

    def test_cut_grows_with_position(self):
        (main, s), tok = _fixture(), CharTok()
        lengths = [len(prefill_committing_value(s, p, s.expected_reasoning[p], tok, main.decode))
                   for p in range(len(s.expected_reasoning))]
        assert lengths == sorted(lengths)
        assert lengths[0] < lengths[-1]

    def test_correct_prefill_is_prefix_of_gold(self):
        (main, s), tok = _fixture(), CharTok()
        gold = s.render(s.expected_reasoning)
        for p in range(len(s.expected_reasoning)):
            assert gold.startswith(prefill_committing_value(s, p, s.expected_reasoning[p], tok, main.decode))

    def test_prefill_decodes_to_the_injected_value(self):
        (main, s), tok = _fixture(), CharTok()
        for p in range(len(s.expected_reasoning)):
            wrong = (s.expected_reasoning[p] + 1) % 4
            decoded = main.decode(prefill_committing_value(s, p, wrong, tok, main.decode))
            assert decoded[p] == wrong
