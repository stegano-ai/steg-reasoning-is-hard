import pytest

from lib.side_tasks import get_side_task
from lib.side_tasks.ball_and_cups import _ball_positions
from lib.side_tasks.rolling_mod_sum import _running_sum_mod


# ── running sum mod computation ──


class TestRunnningSumMod:
    def test_basic_mod3(self):
        assert _running_sum_mod([3, 1, 2], 3) == [0, 1, 0]

    def test_single_element(self):
        assert _running_sum_mod([5], 3) == [2]

    def test_all_zeros(self):
        assert _running_sum_mod([0, 0, 0], 3) == [0, 0, 0]

    def test_mod2(self):
        assert _running_sum_mod([1, 1, 1, 1], 2) == [1, 0, 1, 0]

    def test_accumulates(self):
        assert _running_sum_mod([1, 2, 3, 4], 5) == [1, 3, 1, 0]


# ── messaging side task ──


class TestMessaging:
    def test_loads(self):
        task = get_side_task("messaging", n_samples=100, seq_length=5, vocab_size=3)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_input_reasoning_output_match(self):
        task = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3)
        for item in task.train_data:
            assert item["input"] == item["reasoning"] == item["output"]

    def test_values_in_range(self):
        task = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3)
        for item in task.train_data:
            assert all(0 <= v <= 2 for v in item["input"])

    def test_correct_length(self):
        task = get_side_task("messaging", n_samples=50, seq_length=9, vocab_size=3)
        for item in task.train_data:
            assert len(item["input"]) == len(item["reasoning"]) == len(item["output"]) == 9

    def test_deterministic(self):
        t1 = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3, seed=42)
        t2 = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3, seed=42)
        assert t1.train_data == t2.train_data

    def test_different_seeds(self):
        t1 = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3, seed=1)
        t2 = get_side_task("messaging", n_samples=50, seq_length=5, vocab_size=3, seed=2)
        assert t1.train_data != t2.train_data


# ── rolling mod sum side task ──


class TestRollingModSum:
    def test_loads(self):
        task = get_side_task("rolling_mod_sum", n_samples=100, seq_length=9, operand_range=10, modulus=3)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_reasoning_is_running_sum_mod(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert item["reasoning"] == _running_sum_mod(item["input"], 3)

    def test_output_is_reversed_reasoning(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert item["output"] == list(reversed(item["reasoning"]))

    def test_values_in_range(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert all(0 <= v <= 2 for v in item["reasoning"])
            assert all(0 <= v <= 2 for v in item["output"])

    def test_lengths_match(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=9, operand_range=10, modulus=3)
        for item in task.train_data:
            assert len(item["input"]) == len(item["reasoning"]) == len(item["output"]) == 9

    def test_different_modulus(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=5)
        for item in task.train_data:
            assert all(0 <= v <= 4 for v in item["reasoning"])


# ── ball and cups side task ──


class TestBallAndCups:
    def test_loads(self):
        task = get_side_task("ball_and_cups", n_samples=100, seq_length=9)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_example_from_explanation(self):
        assert _ball_positions([1, 0, 0, 1]) == [1, 2, 3, 4]

    def test_rotate_cycles(self):
        assert _ball_positions([0, 0, 0, 0, 0]) == [1, 2, 3, 4, 0]

    def test_double_swap_is_involution(self):
        assert _ball_positions([1, 1]) == [1, 0]

    def test_output_is_reversed_reasoning(self):
        task = get_side_task("ball_and_cups", n_samples=50, seq_length=5)
        for item in task.train_data:
            assert item["output"] == list(reversed(item["reasoning"]))
            assert item["reasoning"] == _ball_positions(item["input"])

    def test_values_in_range(self):
        task = get_side_task("ball_and_cups", n_samples=50, seq_length=5)
        assert task.alphabet == [0, 1, 2, 3, 4]
        for item in task.train_data:
            assert all(v in (0, 1) for v in item["input"])
            assert all(0 <= v <= 4 for v in item["reasoning"])


# ── limited arithmetic side task ──


class TestLimitedArithmetic:
    def test_loads(self):
        task = get_side_task("limited_arithmetic", n_samples=100, num_steps=3, max_value=20)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_reasoning_length(self):
        task = get_side_task("limited_arithmetic", n_samples=50, num_steps=4, max_value=20)
        for item in task.train_data:
            assert len(item["reasoning"]) == 4

    def test_output_is_final_reasoning_value(self):
        task = get_side_task("limited_arithmetic", n_samples=50, num_steps=3, max_value=20)
        for item in task.train_data:
            assert item["output"] == [item["reasoning"][-1]]

    def test_input_is_string(self):
        task = get_side_task("limited_arithmetic", n_samples=50, num_steps=3, max_value=20)
        for item in task.train_data:
            assert isinstance(item["input"], str)

    def test_values_in_range(self):
        task = get_side_task("limited_arithmetic", n_samples=50, num_steps=3, max_value=20)
        for item in task.train_data:
            assert all(0 <= v <= 19 for v in item["reasoning"])


# ── get_side_task ──


class TestGetSideTask:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown side task type"):
            get_side_task("nonexistent")

    def test_all_tasks_have_name(self):
        tasks = [
            get_side_task("messaging", n_samples=10, seq_length=5, vocab_size=3),
            get_side_task("rolling_mod_sum", n_samples=10, seq_length=5, operand_range=10, modulus=3),
            get_side_task("limited_arithmetic", n_samples=10, num_steps=3, max_value=20),
            get_side_task("ball_and_cups", n_samples=10, seq_length=5),
        ]
        assert [t.name for t in tasks] == ["messaging", "rolling_mod_sum", "limited_arithmetic", "ball_and_cups"]
