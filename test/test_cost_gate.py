from pathlib import Path

import pytest

from lib import cost_estimate


def test_a_yes_proceeds(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    cost_estimate.approve_or_exit()


def test_a_no_aborts_before_anything_is_billed(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit, match="nothing was billed"):
        cost_estimate.approve_or_exit()


def test_an_empty_answer_is_a_no(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(SystemExit):
        cost_estimate.approve_or_exit()


def test_eof_at_the_prompt_is_a_decline_not_a_crash(monkeypatch):
    def eof(_):
        raise EOFError
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(SystemExit, match="nothing was billed"):
        cost_estimate.approve_or_exit()


def test_an_unanswerable_prompt_refuses_rather_than_assuming_yes(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="Refusing to train"):
        cost_estimate.approve_or_exit()


def test_the_dataset_builder_records_eval_prefill():
    """max_stage_cost reads eval_prefill_tokens on every run, so the build must write it."""
    source = Path("experiments/sft/functions/train_sft.py").read_text()
    assert "eval_prefill_tokens" in source


def test_a_dead_price_feed_does_not_kill_a_training_run():
    """train_sft prints the COST line inside a guard; the feed being down must degrade
    to a message, never an exception into the training path."""
    source = Path("experiments/sft/functions/train_sft.py").read_text()
    guarded = source[source.index("stage_cost = cost_estimate.max_stage_cost"):]
    assert "except Exception" in guarded.split("await train.main")[0]
