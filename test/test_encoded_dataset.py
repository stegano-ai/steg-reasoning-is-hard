from lib.encoded_dataset import EncodedDataset, Sample
from lib.main_tasks._standard import StandardSchemeTask
from lib.main_tasks.knapsack import KnapsackTask
from lib.side_tasks import get_side_task


class FakeStdTask(StandardSchemeTask):
    name = "fakestd"

    def __init__(self, scheme):
        super().__init__(scheme)
        self._data = [
            ("q1", "The quick brown fox jumps over the lazy dog while it slowly rains all day long"),
            ("q2", "Hello world this is a simple innocuous cover sentence written for testing purposes here"),
        ]

    def _natural_pool(self, *, train):
        return self._data

    def check_answer(self, prompt, model_output, label):
        return True


def _dataset():
    main = KnapsackTask(scheme="binary_2", n_train=8, n_test=4, n_items=5,
                        capacity=15, max_value=99, max_weight=10, n_candidates=4, seed=1)
    side = get_side_task(type="rolling_mod_sum", n_samples=50, seq_length=4,
                         operand_range=5, modulus=4)
    return main, EncodedDataset(main, side)


class TestEncodedDataset:
    def test_train_and_test_nonempty(self):
        _, ds = _dataset()
        assert len(ds.train()) > 0
        assert len(ds.test()) > 0

    def test_sample_shape(self):
        _, ds = _dataset()
        s = ds.train()[0]
        assert isinstance(s, Sample)
        assert s.main_prompt
        assert len(s.expected_reasoning) == 4
        assert s.alphabet == [0, 1, 2, 3]

    def test_render_roundtrips(self):
        main, ds = _dataset()
        s = ds.train()[0]
        decoded = main.decode(s.render(s.expected_reasoning))
        assert decoded[:len(s.expected_reasoning)] == s.expected_reasoning

    def test_reasoning_is_a_function_of_numbers(self):
        main, ds = _dataset()
        s = ds.train()[0]
        other = [(v + 1) % 4 for v in s.expected_reasoning]
        assert s.render(s.expected_reasoning) != s.render(other)
        assert main.decode(s.render(other))[:len(other)] == other

    def test_same_idx_differs_only_at_changed_position(self):
        _, ds = _dataset()
        s = ds.train()[0]
        a = s.render(s.expected_reasoning)
        flipped = list(s.expected_reasoning)
        flipped[-1] = (flipped[-1] + 1) % 4
        b = s.render(flipped)
        # identical prefix up to the last candidate line (only its bits change)
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        assert i > len(a) // 2


def _std_task_side():
    task = FakeStdTask("capitalization")
    side = get_side_task(type="rolling_mod_sum", n_samples=40, seq_length=3, operand_range=4, modulus=3)
    return task, side


class TestMinPrefix:
    def test_truncates_cover_but_still_roundtrips(self):
        task, side = _std_task_side()
        full = EncodedDataset(task, side).train()[0]
        mini = EncodedDataset(task, side, min_prefix=True).train()[0]
        g_full = full.render(full.expected_reasoning)
        g_min = mini.render(mini.expected_reasoning)
        assert len(g_min) < len(g_full)                                   # cover shortened
        assert task.decode(g_min)[:len(mini.expected_reasoning)] == mini.expected_reasoning  # roundtrips

    def test_gold_shuffled_original_share_truncated_cover(self):
        task, side = _std_task_side()
        s = EncodedDataset(task, side, min_prefix=True).train()[0]
        gold = s.render(s.expected_reasoning)
        shuffled = s.render(list(reversed(s.expected_reasoning)))
        original = s.render([])                                        # encode nothing -> bare cover
        # capitalization only changes case, so the three share one underlying (truncated) cover
        assert gold.lower() == shuffled.lower() == original.lower()
        assert gold != original                                          # gold actually encodes something


class TestNoEncoding:
    def _no_encoding(self):
        task = FakeStdTask(None)
        side = get_side_task(type="rolling_mod_sum", n_samples=40, seq_length=3, operand_range=4, modulus=3)
        return task, EncodedDataset(task, side)

    def test_dataset_nonempty_despite_no_roundtrip(self):
        task, ds = self._no_encoding()
        assert task.encodes is False
        assert len(ds.train()) > 0                                       # roundtrip filter is skipped
        assert len(ds.test()) > 0

    def test_cover_is_left_unencoded(self):
        task, ds = self._no_encoding()
        s = ds.train()[0]
        natural = task._natural_pool(train=True)[s.idx][1]
        assert s.render(s.expected_reasoning) == natural                 # values are not hidden in the text
