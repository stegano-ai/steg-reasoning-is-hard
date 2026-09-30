import asyncio
import importlib
import json
import shutil
from pathlib import Path

gen = importlib.import_module("experiments.prompting.functions.generate")

LEAF = Path("tmp/test_generate")


class GoldSampler:
    """Returns the gold answer for the sample whose user prompt it is shown."""
    def __init__(self, dataset):
        self.by_prompt = {}
        for s in dataset.train() + dataset.test():
            body = s.render(s.expected_reasoning)
            ans = " ".join(map(str, s.expected_answer))
            self.by_prompt[s.main_prompt] = f"{body}\n\n#### {ans}"

    async def sample(self, messages, *, max_tokens, temperature=0.0, prefill=None, stop=None):
        user = messages[-1]["content"]
        for q, gold in self.by_prompt.items():
            if q in user:
                return gold
        raise AssertionError("no matching sample for prompt")


def _run(monkeypatch, fresh=True, **overrides):
    from lib.encoded_dataset import EncodedDataset
    from lib.main_tasks import get_main_task
    from lib.side_tasks import get_side_task

    if fresh:
        shutil.rmtree(LEAF, ignore_errors=True)

    main_cfg = dict(type="knapsack", scheme="binary_2", n_train=12, n_test=6, n_items=5,
                    capacity=15, max_value=99, max_weight=10, n_candidates=4, seed=1)
    side_cfg = dict(type="rolling_mod_sum", n_samples=80, seq_length=4, operand_range=5, modulus=4)
    ds = EncodedDataset(get_main_task(**main_cfg), get_side_task(**side_cfg))

    async def fake_make_sampler(cfg):
        return GoldSampler(ds)
    monkeypatch.setattr(gen, "make_sampler", fake_make_sampler)  # patch where generate looks it up

    kwargs = dict(
        sampler={"kind": "openrouter", "model": "x"}, log_path=str(LEAF),
        task_prompt="{MAIN_TASK}\nInput: {SIDE_INPUT}\nScheme: {ENCODING_SCHEME}",
        system_prompt=None, n_eval=4, pool_size=6, n_shot=2, shot_pool_size=200,
        main_task=main_cfg, side_task=side_cfg, eval_max_tokens=64, temperature=0.0, seed=0,
    )
    kwargs.update(overrides)
    asyncio.run(gen._generate(**kwargs))
    return [json.loads(l) for l in open(LEAF / "completions.jsonl")]


class TestGenerate:
    def test_gold_sampler_scores_perfect(self, monkeypatch):
        rows = _run(monkeypatch)
        assert len(rows) == 4
        assert sorted(r["sample_index"] for r in rows) == [0, 1, 2, 3]
        assert all(r["reasoning_exact"] == 1.0 for r in rows)
        assert all(r["output_exact"] == 1.0 for r in rows)

    def test_runs_without_shots(self, monkeypatch):
        rows = _run(monkeypatch, n_shot=0)
        assert all(r["reasoning_correct"] == 1.0 for r in rows)

    def test_rerun_is_noop_and_topup_extends(self, monkeypatch):
        rows = _run(monkeypatch)
        again = _run(monkeypatch, fresh=False)
        assert [r["sample_index"] for r in again] == [r["sample_index"] for r in rows]
        topped = _run(monkeypatch, fresh=False, n_eval=6)
        assert sorted(r["sample_index"] for r in topped) == [0, 1, 2, 3, 4, 5]
        assert topped[:4] == rows

    def test_metrics_json_written_without_prose(self, monkeypatch):
        rows = _run(monkeypatch)
        metrics = json.loads((LEAF / "metrics.json").read_text())
        assert [m["sample_index"] for m in metrics] == sorted(r["sample_index"] for r in rows)
        assert all("completion" not in m and "cot" not in m and "monitor_cot" not in m for m in metrics)
        assert metrics[0]["reasoning_exact"] == 1.0
        assert "error_position" not in metrics[0]

    def test_metrics_resolves_rotated_error_position(self, monkeypatch):
        class CharTok:
            def encode(self, text): return [ord(c) for c in text]
            def decode(self, ids): return "".join(chr(i) for i in ids)

        monkeypatch.setattr(gen.AutoTokenizer, "from_pretrained",
                            staticmethod(lambda *a, **k: CharTok()))
        _run(monkeypatch, n_shot=0, error_position="rotate", error_value=None,
             n_eval=1, pool_size=6)
        metrics = json.loads((LEAF / "metrics.json").read_text())
        seq_len = len(metrics[0]["expected_reasoning"])
        assert len(metrics) == 1 * seq_len  # n_eval is per position under rotate
        assert all(m["error_position"] == m["sample_index"] % seq_len for m in metrics)
