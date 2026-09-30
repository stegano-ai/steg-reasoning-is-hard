import pytest
from unittest.mock import patch, AsyncMock

from lib.main_tasks import get_main_task
from lib.main_tasks.aqua_rat import _extract_final as aqua_extract, _format_question, _format_answer
from lib.main_tasks.wildchat import WildChatTask


# ── AQUA-RAT ──


class TestAquaRatFormatQuestion:
    def test_options_appended(self):
        q = _format_question("What is 2+2?", ["A)4", "B)5", "C)6", "D)7", "E)8"])
        assert "What is 2+2?" in q
        assert "A)4" in q
        assert "E)8" in q

    def test_options_on_separate_lines(self):
        q = _format_question("Q?", ["A)1", "B)2", "C)3", "D)4", "E)5"])
        lines = q.split("\n")
        option_lines = [l for l in lines if l.startswith(("A)", "B)", "C)", "D)", "E)"))]
        assert len(option_lines) == 5


class TestAquaRatFormatAnswer:
    def test_strips_answer_line_and_appends_marker(self):
        result = _format_answer("step 1\nstep 2\nAnswer: C", "C")
        assert result.endswith("#### C")
        assert "Answer: C" not in result
        assert "step 2" in result

    def test_strips_bare_letter(self):
        result = _format_answer("reasoning\nE", "E")
        assert result.endswith("#### E")
        assert result.count("E") == 1  # only in the #### line

    def test_always_drops_last_line(self):
        result = _format_answer("line 1\nline 2\nsome final line with math", "A")
        assert "some final line with math" not in result
        assert result.endswith("#### A")


class TestAquaRatExtractFinal:
    def test_basic(self):
        assert aqua_extract("reasoning\n#### C") == "C"

    def test_case_insensitive(self):
        assert aqua_extract("#### c") == "C"

    def test_no_marker(self):
        assert aqua_extract("no answer") == ""


@pytest.mark.slow
class TestAquaRatLoad:
    def test_loads(self):
        task = get_main_task("aqua_rat", scheme="capitalization")
        assert len(task._natural_pool(train=True)) > 0
        assert len(task._natural_pool(train=False)) > 0

    def test_check_answer_on_real_data(self):
        task = get_main_task("aqua_rat", scheme="capitalization")
        q, a = task._natural_pool(train=True)[0]
        assert task.check_answer(q, a, a)
        assert not task.check_answer(q, "#### Z", a)


# ── WildChat ──


@pytest.mark.slow
class TestWildchatLoad:
    def test_loads(self):
        task = get_main_task("wildchat", scheme="capitalization", n_train=10, n_test=5, seed=42)
        assert len(task._natural_pool(train=True)) == 10
        assert len(task._natural_pool(train=False)) == 5

    def test_no_overlap(self):
        task = get_main_task("wildchat", scheme="capitalization", n_train=10, n_test=5, seed=42)
        train_q = {q for q, _ in task._natural_pool(train=True)}
        test_q = {q for q, _ in task._natural_pool(train=False)}
        assert train_q.isdisjoint(test_q)

    def test_deterministic(self):
        t1 = get_main_task("wildchat", scheme="capitalization", n_train=5, n_test=3, seed=42)
        t2 = get_main_task("wildchat", scheme="capitalization", n_train=5, n_test=3, seed=42)
        assert t1._natural_pool(train=True) == t2._natural_pool(train=True)


class TestWildchatCheckAnswer:
    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_good_response_passes(self, mock_monitor):
        mock_monitor.return_value = {"good": "yes"}
        task = WildChatTask.__new__(WildChatTask)  # bypass __init__ to avoid streaming load
        assert await task.check_answer("what is python?", "Python is a language.", "") is True

    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_bad_response_fails(self, mock_monitor):
        mock_monitor.return_value = {"good": "no"}
        task = WildChatTask.__new__(WildChatTask)
        assert await task.check_answer("what is python?", "asdfghjkl", "") is False

    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_missing_key_fails(self, mock_monitor):
        mock_monitor.return_value = {}
        task = WildChatTask.__new__(WildChatTask)
        assert await task.check_answer("q", "a", "") is False

    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_case_insensitive(self, mock_monitor):
        task = WildChatTask.__new__(WildChatTask)
        mock_monitor.return_value = {"good": "Yes"}
        assert await task.check_answer("q", "a", "") is True
        mock_monitor.return_value = {"good": "YES"}
        assert await task.check_answer("q", "a", "") is True

    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_prompt_passed_to_monitor(self, mock_monitor):
        mock_monitor.return_value = {"good": "yes"}
        task = WildChatTask.__new__(WildChatTask)
        await task.check_answer("my question", "my response", "")
        task_arg = mock_monitor.call_args[0][0]
        assert task_arg == "my question"

    @pytest.mark.asyncio
    @patch("lib.main_tasks.wildchat._monitor_cot_async", new_callable=AsyncMock)
    async def test_uses_correct_model(self, mock_monitor):
        mock_monitor.return_value = {"good": "yes"}
        task = WildChatTask.__new__(WildChatTask)
        await task.check_answer("q", "a", "")
        model_arg = mock_monitor.call_args[0][2]
        assert model_arg == "openai/gpt-5-mini-2025-08-07"


# ── get_main_task ──


class TestGetMainTask:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown main task type"):
            get_main_task("nonexistent")

    @pytest.mark.slow
    def test_all_tasks_have_name(self):
        for name in ["aqua_rat"]:
            task = get_main_task(name, scheme="capitalization")
            assert task.name == name


# ── encode/decode roundtrip via the public API ──


_FAKE_SIDE = [
    {"input": [0, 0, 0], "reasoning": [0, 1, 2], "output": [0, 1, 2]},
    {"input": [1, 0, 1], "reasoning": [2, 1, 0], "output": [2, 1, 0]},
    {"input": [0, 1, 0], "reasoning": [1, 1, 2], "output": [1, 1, 2]},
]


def _fake_side():
    from lib.side_tasks.base import SideTaskDefinition
    return SideTaskDefinition(name="fake", train_data=_FAKE_SIDE, test_data=_FAKE_SIDE,
                              explanation="fake", alphabet=[0, 1, 2])


class TestSideTasks:
    """End-to-end roundtrip via EncodedDataset. Uses fake side data so no network for side."""

    @pytest.mark.slow
    def test_aqua_rat_capitalization(self):
        from lib.encoded_dataset import EncodedDataset
        task = get_main_task("aqua_rat", scheme="capitalization")
        samples = EncodedDataset(task, _fake_side()).train()
        assert len(samples) > 0
        s = samples[0]
        assert task.decode(s.render(s.expected_reasoning))[:len(s.expected_reasoning)] == s.expected_reasoning

    @pytest.mark.slow
    def test_aqua_rat_whitespace(self):
        from lib.encoded_dataset import EncodedDataset
        task = get_main_task("aqua_rat", scheme="whitespace")
        samples = EncodedDataset(task, _fake_side()).train()
        assert len(samples) > 0
        s = samples[0]
        assert task.decode(s.render(s.expected_reasoning))[:len(s.expected_reasoning)] == s.expected_reasoning
