import pytest

from lib import cost_estimate
from lib.cost_estimate import format_stage_cost, max_stage_cost

# A run as its dataset build would report it: 1600 datums of 300 tokens, an NLL split of
# 3 * 12k tokens, 10k of eval prefill.
STATS = {
    "train_datums": 1600,
    "train_tokens_mean": 300.0,
    "nll_test_tokens": 12_000,
    "nll_shuffled_tokens": 12_000,
    "nll_original_tokens": 12_000,
    "eval_prefill_tokens": 10_000,
}
PRICE = {"train": 2.0, "sample": 4.0, "prefill": 1.0}


@pytest.fixture(autouse=True)
def fixed_price(monkeypatch):
    """The arithmetic is under test, not the live feed — that is the slow test below."""
    monkeypatch.setattr(cost_estimate, "prices", lambda: {"m": PRICE})


def estimate(**overrides):
    kwargs = dict(stats=STATS, model_name="m", sft={"num_epochs": 1, "eval_every": 10},
                  batch_size=16, n_eval=40, eval_max_tokens=1000,
                  infrequent_eval_every=100)
    kwargs.update(overrides)
    return max_stage_cost(**kwargs)


def test_steps_come_from_the_surviving_datums_not_the_config():
    # 1600 datums / batch 16 = 100 steps; the config's n_train never enters.
    assert estimate()["steps"] == 100


def test_max_steps_caps_the_epoch_count():
    est = estimate(sft={"num_epochs": 100, "eval_every": 10, "max_steps": 250})
    assert est["steps"] == 250


def test_absent_max_steps_means_uncapped():
    est = estimate(sft={"num_epochs": 3, "eval_every": 10})
    assert est["steps"] == 300


def test_train_tokens_are_steps_times_batch_times_measured_mean():
    est = estimate()
    assert est["train_tokens"] == 100 * 16 * 300.0
    assert est["train_cost"] == pytest.approx(est["train_tokens"] / 1e6 * PRICE["train"])


def test_nll_runs_every_eval_every_and_bills_at_the_training_rate():
    est = estimate()
    assert est["nll_tokens"] == (100 // 10 + 1) * 36_000
    assert est["nll_cost"] == pytest.approx(est["nll_tokens"] / 1e6 * PRICE["train"])


def test_sampling_is_priced_at_the_eval_max_tokens_ceiling():
    est = estimate()
    n_gen = 100 // 100 + 1
    assert est["sampled_tokens"] == n_gen * 40 * 1000
    assert est["prefill_tokens"] == n_gen * 10_000


def test_eval_every_zero_disables_the_nll_term():
    est = estimate(sft={"num_epochs": 1, "eval_every": 0})
    assert est["nll_tokens"] == 0


def test_no_generation_eval_costs_no_sampling():
    est = estimate(n_eval=0, eval_max_tokens=0, infrequent_eval_every=None)
    assert est["sampled_tokens"] == 0 and est["prefill_tokens"] == 0


def test_unpriced_model_raises_and_names_the_feed():
    with pytest.raises(KeyError, match="is not priced in"):
        estimate(model_name="some/unlisted-model")


def test_the_stage_line_carries_the_total_of_its_parts():
    est = estimate()
    line = format_stage_cost("results/x/stage1", est)
    total = est["train_cost"] + est["nll_cost"] + est["sample_cost"] + est["prefill_cost"]
    assert "stage1" in line and cost_estimate._money(total) in line


@pytest.mark.slow
def test_live_feed_parses_and_prices_are_positive(monkeypatch):
    monkeypatch.undo()
    cost_estimate.prices.cache_clear()
    table = cost_estimate.prices()
    assert len(table) > 5
    for model, price in table.items():
        assert set(price) == {"train", "sample", "prefill"}, model
        assert all(v > 0 for v in price.values()), (model, price)


@pytest.mark.slow
def test_every_model_named_in_a_config_is_priced_by_the_feed(monkeypatch):
    import glob

    import yaml

    monkeypatch.undo()
    cost_estimate.prices.cache_clear()
    named = set()

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "model_name" and isinstance(value, str):
                    named.add(value)
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    for path in glob.glob("experiments/sft/configs/**/*.yaml", recursive=True):
        with open(path) as f:
            for doc in yaml.safe_load_all(f):
                collect(doc)
    assert named, "no model_name found in any config"
    table = cost_estimate.prices()
    assert named <= set(table), f"unpriced models in configs: {sorted(named - set(table))}"


def test_the_run_table_is_plain_and_aligned_when_not_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    est = estimate()
    out = cost_estimate.format_run_cost("run", "m", [est, est])
    assert "\033" not in out                      # no ANSI escapes into logs/pipes
    lines = out.splitlines()
    assert "Max Tinker cost for run:" in lines[0] and "(m)" in lines[0]
    header, *rows = lines[1:]
    assert header.split() == ["stage", "steps", "train", "nll", "sampling", "total"]
    assert [r.split()[0] for r in rows] == ["1", "2", "all"]
    assert rows[-1].split()[1] == "200"           # steps summed across stages


def test_multi_run_invocations_close_with_a_grand_total(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    line = cost_estimate.format_grand_total({"a": 10.0, "b": 15.5})
    assert line == "Total for 2 runs: $26"
