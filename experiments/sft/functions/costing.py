"""Fast pre-run cost ceiling: sample the datasets into a temp dir, price the run.

Runs before the scaffold opens wandb and before any Tinker client exists — declining at
the gate leaves nothing behind: no results directory, no wandb run, nothing billed.

Fast because almost nothing needs building: the NLL and eval splits are n_test/n_eval
samples and are built exactly; only the train-token mean is estimated, from the first
SAMPLE_N kept datums — the keep order is a seeded shuffle, so they are a uniform draw
from the same pool the real build will keep.
"""

import contextlib
import functools
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

from lib.cost_estimate import max_stage_cost
from lib.main_tasks import get_main_task
from lib.side_tasks import get_side_task

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .train_sft import build_datasets

SAMPLE_N = 64


# Runs of one sweep often share their model or tasks; loading the tokenizer or the HF
# dataset once per run would dominate the whole pass.
@functools.cache
def _renderer(model_name: str, renderer_name: str | None):
    tokenizer = get_tokenizer(model_name)
    name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(name, tokenizer=tokenizer)


@functools.cache
def _tasks(main_json: str, side_json: str):
    return get_main_task(**json.loads(main_json)), get_side_task(**json.loads(side_json))


def _sampled_stats(kwargs: dict, tmp: str) -> dict:
    """token_stats for a SAMPLE_N-train build of this run, written under tmp."""
    task, side = _tasks(json.dumps(kwargs["main_task"], sort_keys=True),
                        json.dumps(kwargs["side_task"], sort_keys=True))
    renderer = _renderer(kwargs["model_name"], kwargs["renderer_name"])
    build_datasets(renderer, task, side,
                   batch_size=kwargs["batch_size"], task_prompt=kwargs["task_prompt"],
                   n_train=min(kwargs["n_train"], SAMPLE_N), n_test=kwargs["n_test"],
                   n_eval=kwargs["n_eval"], max_tokens=kwargs["max_tokens"],
                   min_prefix=kwargs["min_prefix"],
                   closed_prefix_targets=kwargs["closed_prefix_targets"],
                   seed=kwargs["seed"], log_path=tmp)
    return json.loads((Path(tmp) / "token_stats.json").read_text())


def stage_estimates(config) -> list[dict]:
    """One max_stage_cost dict per resolved Config, sampled, nothing kept."""
    kwargs = config.function_kwargs
    progress = f"  costing {config.name} ..."
    print(f"\r{progress}", end="", flush=True, file=sys.stderr)
    # The sampled build prints its usual filter/TOKEN_STATS lines and the cookbook logs
    # at INFO; during costing all of that is noise around a one-table answer.
    with tempfile.TemporaryDirectory() as tmp, \
            contextlib.redirect_stdout(io.StringIO()):
        logging.disable(logging.INFO)
        try:
            stats = _sampled_stats(kwargs, tmp)
        finally:
            logging.disable(logging.NOTSET)
    # The real build keeps exactly n_train datums (build_datasets raises otherwise);
    # only the per-datum token mean comes from the sample.
    stats["train_datums"] = kwargs["n_train"]
    print("\r" + " " * len(progress) + "\r", end="", flush=True, file=sys.stderr)
    return [max_stage_cost(
        stats=stats, model_name=kwargs["model_name"], sft=kwargs["sft"],
        batch_size=kwargs["batch_size"], n_eval=kwargs["n_eval"],
        eval_max_tokens=kwargs["eval_max_tokens"] if kwargs["n_eval"] > 0 else 0,
        infrequent_eval_every=kwargs["infrequent_eval_every"],
    )]
