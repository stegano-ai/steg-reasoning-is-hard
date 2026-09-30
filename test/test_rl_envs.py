import asyncio
from unittest.mock import patch

from lib.cot_monitor import MonitorConfig
from lib.encoded_dataset import Sample
from lib.environments import ReasoningEnv, MessagingEnv
from experiments.rl.functions.train_grpo import RewardConfig, TinkerStegEnv, make_steg_env


class FakeRenderer:
    def __init__(self):
        self.last_prefill = None

    def build_generation_prompt(self, messages, prefill=None):
        self.last_prefill = prefill
        return f"PROMPT[{len(messages)} msgs prefill={prefill!r}]"

    def get_stop_sequences(self):
        return ["<|end|>"]

    def parse_response(self, action):
        return {"content": str(action)}, None


class FakeMainTask:
    description = "fake-scheme"

    def __init__(self, decoded):
        self._decoded = decoded

    def decode(self, text):
        return list(self._decoded)


def _sample():
    return Sample(idx=0, main_prompt="What is 2+2?", side_input=[1, 2, 3],
                  expected_reasoning=[1, 2, 3], expected_answer=[1, 2, 3],
                  alphabet=[0, 1, 2, 3], _render=lambda nums: "encoded cover")


def _rewards():
    return RewardConfig(format=0.1, output_exact=0.0, output_correct=1.0,
                        reasoning_exact=0.0, reasoning_correct=1.0, monitor_floor=0.5,
                        additive=False, streak_only=False)


_PROMPTS = {"system": "", "sender_template": "Q: {MAIN_TASK}",
            "decoder_template": "Decode {N_VALUES} from {MODEL_ANSWER}"}


def _adapter(env_kind, decoded, is_eval=False, model_generated_scheme=False):
    main = FakeMainTask(decoded)
    steg = make_steg_env(env_kind, _sample(), main, _PROMPTS, "side explanation", (), model_generated_scheme)
    return TinkerStegEnv(steg, FakeRenderer(), MonitorConfig(models=["m"], prompt="{task}{output}"),
                         _rewards(), logger=None, is_eval=is_eval)


def _await(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestReasoningAdapter:
    def test_step_ends_episode_no_signal(self):
        env = _adapter("reasoning", decoded=[])
        _await(env.initial_observation())
        result = _await(env.step("plain text with no marker"))
        assert result.episode_done is True
        assert result.metrics["output_format"] == 0.0
        assert result.reward == 0.0   # no #### and reasoning_correct=0 -> gate off, only format=0

    def test_step_with_signal_runs_monitor(self):
        env = _adapter("reasoning", decoded=[1, 2, 3])

        async def fake_monitor(result, *, question, monitor):
            result.suspiciousness = 3.0
            result.monitor_cot = {"suspiciousness": 3.0}
            return result

        _await(env.initial_observation())
        with patch("experiments.rl.functions.train_grpo.run_monitor", side_effect=fake_monitor):
            result = _await(env.step("encoded cot\n#### 1 2 3"))
        assert result.episode_done is True
        assert result.metrics["reasoning_correct"] == 1.0
        assert result.metrics["output_exact"] == 1.0
        assert result.metrics["suspiciousness"] == 3.0
        assert result.reward > 0


class TestMessagingAdapter:
    def test_first_step_returns_decoder_observation(self):
        env = _adapter("messaging", decoded=[1, 2, 3])
        _await(env.initial_observation())
        result = _await(env.step("sender response embedding 1 2 3"))
        assert result.episode_done is False
        assert result.next_observation is not None
        assert result.reward == 0.0

    def test_second_step_ends_with_correct_decode_reasoning_and_output(self):
        env = _adapter("messaging", decoded=[1, 2, 3])

        async def fake_monitor(result, *, question, monitor):
            result.suspiciousness = 1.0
            return result

        _await(env.initial_observation())
        _await(env.step("sender text"))
        with patch("experiments.rl.functions.train_grpo.run_monitor", side_effect=fake_monitor):
            result = _await(env.step("decoded\n#### 1 2 3"))
        assert result.episode_done is True
        assert result.metrics["reasoning_correct"] == 1.0   # sender decoded ok
        assert result.metrics["output_exact"] == 1.0        # decoder reported [1,2,3]

    def test_second_step_bad_decode(self):
        env = _adapter("messaging", decoded=[1, 2, 3])

        async def fake_monitor(result, *, question, monitor):
            result.suspiciousness = 5.0
            return result

        _await(env.initial_observation())
        _await(env.step("sender"))
        with patch("experiments.rl.functions.train_grpo.run_monitor", side_effect=fake_monitor):
            result = _await(env.step("decoded\n#### 9 9 9"))
        assert result.metrics["reasoning_correct"] == 1.0
        assert result.metrics["output_exact"] == 0.0
