import asyncio

from lib.encoded_dataset import EncodedDataset
from lib.environments import ReasoningEnv, MessagingEnv, run_inference
from lib.main_tasks.knapsack import KnapsackTask
from lib.prefill import prefill_committing_value
from lib.side_tasks import get_side_task


def _await(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class CharTok:
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids): return "".join(chr(i) for i in ids)


class ScriptedSampler:
    def __init__(self, responses):
        self.responses, self.i, self.seen = list(responses), 0, []

    async def sample(self, messages, *, max_tokens, temperature=0.0, prefill=None, stop=None):
        self.seen.append((messages, prefill))
        r = self.responses[self.i]; self.i += 1
        return r


def _fixture():
    main = KnapsackTask(scheme="binary_2", n_train=8, n_test=4, n_items=5,
                        capacity=15, max_value=99, max_weight=10, n_candidates=4, seed=1)
    side = get_side_task(type="rolling_mod_sum", n_samples=50, seq_length=4,
                         operand_range=5, modulus=4)
    sample = EncodedDataset(main, side).train()[0]
    return main, sample


_TASK_PROMPT = "{MAIN_TASK}\nSide input: {SIDE_INPUT}\nScheme: {ENCODING_SCHEME}"


class TestReasoningEnv:
    def test_gold_completion_scores_perfect(self):
        main, s = _fixture()
        gold = f"{s.render(s.expected_reasoning)}\n\n#### {' '.join(map(str, s.expected_answer))}"
        env = ReasoningEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x")
        result = _await(run_inference(env, ScriptedSampler([gold]), max_tokens=64))
        assert result.reasoning_exact == 1.0
        assert result.output_exact == 1.0

    def test_single_turn_then_done(self):
        main, s = _fixture()
        env = ReasoningEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x")
        assert env.next_turn() is not None
        env.submit("anything")
        assert env.done
        assert env.next_turn() is None

    def test_opening_messages_include_shots(self):
        main, s = _fixture()
        shots = EncodedDataset(main, get_side_task(
            type="rolling_mod_sum", n_samples=50, seq_length=4, operand_range=5, modulus=4)).train()[1:3]
        env = ReasoningEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x", shots=shots)
        msgs = env.next_turn().messages
        # 2 shots -> 2 (user, assistant) pairs + final user
        assert sum(m["role"] == "assistant" for m in msgs) == 2
        assert msgs[-1]["role"] == "user"

    def test_error_prefill_is_used_as_turn_prefill(self):
        main, s = _fixture()
        tok = CharTok()
        wrong = (s.expected_reasoning[1] + 1) % 4
        env = ReasoningEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x",
                        error_position=1, error_value=wrong, tokenizer=tok)
        turn = env.next_turn()
        assert turn.prefill == prefill_committing_value(s, 1, wrong, tok, main.decode)
        env.submit(" continuation")
        assert env._completion == turn.prefill + " continuation"


class TestMessagingEnv:
    def test_two_turns_sender_then_decoder(self):
        main, s = _fixture()
        sender = s.render(s.expected_reasoning)
        decoder = f"#### {' '.join(map(str, s.expected_answer))}"
        env = MessagingEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x",
                           decoder_template="decode: {MODEL_ANSWER} ({N_VALUES})")
        sampler = ScriptedSampler([sender, decoder])
        result = _await(run_inference(env, sampler, max_tokens=64))
        # sender turn carries no prefill; decoder turn references the sender answer
        assert len(sampler.seen) == 2
        assert "decode:" in sampler.seen[1][0][0]["content"]
        assert result.reasoning_exact == 1.0          # decoded from the sender cover
        assert result.output_correct == 1.0           # decoder reported the values


_SCHEME_REQ = "Design a scheme for {N_VALUES} ints in [0,{MAX_VALUE}]: {MAIN_TASK}"


class TestModelDesignedScheme:
    """model_generated_scheme=True adds an opening design turn to either task mode."""

    def test_reasoning_with_model_scheme_two_turns(self):
        main, s = _fixture()
        env = ReasoningEnv(s, main, task_prompt=_TASK_PROMPT, side_explanation="x",
                        model_generated_scheme=True, scheme_request_template=_SCHEME_REQ)
        gold = f"{s.render(s.expected_reasoning)}\n\n#### {' '.join(map(str, s.expected_answer))}"
        sampler = ScriptedSampler(["my scheme: line b -> bit b", gold])
        result = _await(run_inference(env, sampler, max_tokens=64))
        assert len(sampler.seen) == 2                                  # design turn + answer turn
        assert "Design a scheme" in sampler.seen[0][0][-1]["content"]  # turn 1 = design (no numbers)
        assert any(m.get("content") == "my scheme: line b -> bit b" for m in sampler.seen[1][0])  # scheme in history
        assert result.output_exact == 1.0
        assert result.reasoning_format is None   # model-designed scheme -> no mechanical reasoning signal

    def test_messaging_with_model_scheme_three_turns(self):
        main, s = _fixture()
        env = MessagingEnv(s, main, task_prompt="encode {VALUES} with {ENCODING_SCHEME}",
                           side_explanation="x", model_generated_scheme=True, scheme_request_template=_SCHEME_REQ,
                           decoder_template="scheme={SCHEME} ans={MODEL_ANSWER} recover {N_VALUES}")
        scheme = "put bit b on line b"
        sender = s.render(s.expected_reasoning)
        decoder = f"#### {' '.join(map(str, s.expected_answer))}"
        sampler = ScriptedSampler([scheme, sender, decoder])
        result = _await(run_inference(env, sampler, max_tokens=64))
        assert len(sampler.seen) == 3                                  # design + sender + decoder
        assert any(m.get("content") == scheme for m in sampler.seen[1][0])   # scheme in sender history
        assert scheme in sampler.seen[2][0][0]["content"]              # decoder shown the scheme
        assert result.output_correct == 1.0                            # decoder recovered the values
        assert result.reasoning_format is None   # model-designed scheme -> no mechanical reasoning signal
