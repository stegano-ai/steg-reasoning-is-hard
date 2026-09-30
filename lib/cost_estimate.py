"""Measured Tinker cost ceiling for an SFT run, gated just before training spends it.

Ported from steg-transfer-learning. The bill has three token line items — Training
Token, Sampler Sample, Sampler Prefill — plus checkpoint storage. Training Token is the
one that matters, and it covers more than the training batches: the cookbook's supervised
loop runs its evaluators via `training_client.forward_async`, and those forward passes
bill at the training rate. Three NLL evaluators run per eval (the test split, plus the
shuffled and original baselines built in train_sft.build_datasets), each over the full
`n_test` split:

    billed_train_tokens = steps * batch_size * mean_datum_tokens          # forward+backward
                        + (steps // eval_every + 1) * (nll datum tokens)

Validated in steg-transfer-learning against the 2026-07-29 billing rows: predicted vs
actual was 184.0M vs 184.3M (gpt-oss-120b), 155.5M vs 154.9M (Nemotron-3-Super), 143.2M
vs 148.1M (Qwen3.6-35B-A3B).

The NLL term is why an `n_test` bump is expensive: at n_test=1000 and 1000 steps it is
~20x the training term, so it dominates the run.

Every token count is measured: the run has already built its datasets by the time the
ceiling is computed, so tokens-per-datum, the NLL datum sizes and the eval prefill all
come from token_stats.json — the real tokenizer, prompt template, scheme and min-prefix
search. Only two numbers are assumptions, both upper bounds, which is what makes the
total a ceiling: the run never early-stops, and every sampled completion runs to
eval_max_tokens.
"""

import functools
import math
import sys
from pathlib import Path

import requests

PRICES_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


@functools.cache
def prices() -> dict[str, dict[str, float]]:
    """Per-model $/M-token rates, live from the Tinker docs feed.

    The feed's figures are what is actually billed — the limited-time discount several models
    carry is already applied, with the pre-discount figures kept alongside in `original_*`.
    Fetched once per process; nothing is cached to disk, so a stale price is never used.
    """
    rows = requests.get(PRICES_URL, timeout=30).json()
    return {
        row["tinker_id"]: {k: float(row[k].lstrip("$").replace(",", ""))
                           for k in ("train", "sample", "prefill")}
        for row in rows
    }


def _price(model_name: str) -> dict:
    table = prices()
    if model_name not in table:
        raise KeyError(f"{model_name!r} is not priced in {PRICES_URL}; "
                       f"known ids: {sorted(table)}")
    return table[model_name]


def _money(x: float) -> str:
    return f"${x:,.2f}" if x < 10 else f"${x:,.0f}"


def max_stage_cost(*, stats: dict, model_name: str, sft: dict, batch_size: int,
                   n_eval: int, eval_max_tokens: int,
                   infrequent_eval_every: int | None) -> dict:
    """Ceiling for one run, from the token_stats its own dataset build just measured."""
    steps = math.ceil(stats["train_datums"] / batch_size) * sft["num_epochs"]
    if sft.get("max_steps") is not None:
        steps = min(steps, sft["max_steps"])

    n_freq = steps // sft["eval_every"] + 1 if sft["eval_every"] else 0
    n_infreq = steps // infrequent_eval_every + 1 if infrequent_eval_every else 0

    train_tokens = steps * batch_size * stats["train_tokens_mean"]
    nll_tokens = n_freq * sum(stats[k] for k in ("nll_test_tokens", "nll_shuffled_tokens",
                                                 "nll_original_tokens"))
    prefill_tokens = n_infreq * stats["eval_prefill_tokens"]
    sampled_tokens = n_infreq * n_eval * eval_max_tokens

    price = _price(model_name)
    return {
        "steps": steps, "train_tokens": train_tokens, "nll_tokens": nll_tokens,
        "sampled_tokens": sampled_tokens, "prefill_tokens": prefill_tokens,
        # NLL forward passes bill at the training rate; they are kept apart so the breakdown
        # can show what the held-out evals cost on their own — the term an n_test bump inflates.
        "train_cost": train_tokens / 1e6 * price["train"],
        "nll_cost": nll_tokens / 1e6 * price["train"],
        "sample_cost": sampled_tokens / 1e6 * price["sample"],
        "prefill_cost": prefill_tokens / 1e6 * price["prefill"],
    }


def format_stage_cost(log_path: str, estimate: dict) -> str:
    total = sum(estimate[k] for k in
                ("train_cost", "nll_cost", "sample_cost", "prefill_cost"))
    return (f"COST ceiling for {Path(log_path).name}: {_money(total)} over "
            f"{estimate['steps']} steps "
            f"(train {_money(estimate['train_cost'])}, "
            f"nll {_money(estimate['nll_cost'])}, "
            f"sampling {_money(estimate['sample_cost'] + estimate['prefill_cost'])})")


def approve_or_exit() -> None:
    """Block until the user approves the ceiling just printed. Exits if they don't.

    Called before the scaffold opens wandb and before any Tinker client exists, so declining
    leaves nothing behind — no results directory, no wandb run, nothing billed.
    """
    if not sys.stdin.isatty():
        sys.exit("Refusing to train without approval: stdin is not a terminal, so the "
                 "y/n prompt cannot be answered. Drop -p to run without the gate.")
    try:
        answer = input("Proceed? [y/N] ")
    except EOFError:  # Ctrl-D at the prompt is a decline, not a crash
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        sys.exit("Aborted before training; nothing was billed.")


def format_run_cost(name: str, model: str, estimates: list[dict]) -> str:
    """An aligned per-stage table with a bold total — the one thing -p exists to show.

    sampling = sample + prefill; train and nll both bill at the training rate but are
    split so the term an n_test bump inflates is visible on its own.
    """
    bold, dim, end = ("\033[1m", "\033[2m", "\033[0m") if sys.stdout.isatty() else ("",) * 3
    rows = [(str(i), f"{e['steps']:,}", _money(e["train_cost"]), _money(e["nll_cost"]),
             _money(e["sample_cost"] + e["prefill_cost"]),
             _money(sum(e[k] for k in ("train_cost", "nll_cost", "sample_cost", "prefill_cost"))))
            for i, e in enumerate(estimates, 1)]
    total = run_total(estimates)
    head = ("stage", "steps", "train", "nll", "sampling", "total")
    body = rows + [("all", f"{sum(e['steps'] for e in estimates):,}", "", "", "", _money(total))]
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(6)]

    def line(cells, style=""):
        text = "  ".join(c.rjust(w) for c, w in zip(cells, widths))
        return f"  {style}{text}{end if style else ''}"

    return "\n".join([
        f"{bold}Max Tinker cost for {name}: {_money(total)}{end}  {dim}({model}){end}",
        line(head, dim),
        *[line(r) for r in rows],
        line(body[-1], bold),
    ])


def run_total(estimates: list[dict]) -> float:
    return sum(e[k] for e in estimates
               for k in ("train_cost", "nll_cost", "sample_cost", "prefill_cost"))


def format_grand_total(per_run: dict[str, float]) -> str:
    """Closing line for a multi-run invocation: what saying y commits to in total."""
    bold, end = ("\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "")
    return (f"{bold}Total for {len(per_run)} runs: "
            f"{_money(sum(per_run.values()))}{end}")


def resolve_configs(config_path: str) -> list:
    """The resolved Configs a `-c <path>` would run, in the order they would run."""
    from research_scaffold.config_tools import (detect_config_type, load_config,
                                                load_dict_from_yaml, load_meta_config,
                                                process_meta_config)
    config_type = detect_config_type(load_dict_from_yaml(config_path))
    if config_type == "meta":
        return process_meta_config(load_meta_config(config_path))
    if config_type == "sweep":
        return []  # a wandb sweep has no fixed experiment list to price
    return [load_config(config_path)]
