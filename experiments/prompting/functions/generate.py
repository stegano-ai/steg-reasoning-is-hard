import asyncio
import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from lib.cot_monitor import MonitorConfig
from lib.encoded_dataset import EncodedDataset
from lib.environments import MessagingEnv, ReasoningEnv, run_inference
from lib.main_tasks import get_main_task
from lib.prompting import remap_side_input, sample_shots
from lib.sampling import make_sampler
from lib.side_tasks import get_side_task

# The dataset build (cover load + per-sample encode/decode round-trip filtering) depends only
# on the task/data config, not the model — so cells that share it (e.g. every model for one
# scheme×task) build once and reuse, under a lock so concurrent cells single-flight the build.
_PREP_CACHE: dict[str, tuple] = {}
_PREP_LOCK = threading.Lock()

# Everything else in the config defines what a datapoint IS; n_eval only says how many.
_IDENTITY_EXCLUDE = ("n_eval", "log_path")


def _identity(config: dict) -> dict:
    return {k: v for k, v in config.items() if k not in _IDENTITY_EXCLUDE}


async def _generate(**config):
    """Idempotent datapoint generation: the cell folder (log_path) holds config.yaml +
    completions.jsonl; reruns generate only the sample indices in [0, n_eval) not already
    present (top-up and drop-refill). A config that differs in anything but n_eval crashes —
    delete the folder to regenerate."""
    leaf = Path(config["log_path"])
    config_path = leaf / "config.yaml"
    rows_path = leaf / "completions.jsonl"
    if config_path.exists():
        old, new = _identity(yaml.safe_load(config_path.read_text())), _identity(config)
        if old != new:
            changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
            raise RuntimeError(f"{config_path} does not match the requested config "
                               f"(differs in {changed}); delete {leaf} to regenerate")
    done = {json.loads(l)["sample_index"] for l in open(rows_path)} if rows_path.exists() else set()
    n = config["n_eval"]
    if config.get("error_position") == "rotate":  # n_eval is per injected position
        n *= config["side_task"]["seq_length"]
    targets = [i for i in range(n) if i not in done]
    print(f"{leaf}: {len(done)} datapoints present, generating {len(targets)}", flush=True)
    leaf.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if targets:
        await _generate_rows(targets, **config)
    write_metrics(leaf, config)


_PROSE = ("cot", "completion", "monitor_cot", "prompt")


def write_metrics(leaf: Path, config: dict) -> None:
    """metrics.json next to completions.jsonl: every score and int-list per row, no prose.
    For error-injection cells the per-row injected position is resolved (rotate -> int)."""
    rows_path = leaf / "completions.jsonl"
    if not rows_path.exists():
        return
    err = config.get("error_position")
    rows = []
    for line in open(rows_path):
        r = json.loads(line)
        mc = r.get("monitor_cot")
        r["monitor_cost"] = mc.get("cumulative_cost") if isinstance(mc, dict) else None
        for k in _PROSE:
            r.pop(k, None)
        if err is not None:
            r["error_position"] = r["sample_index"] % len(r["expected_reasoning"]) if err == "rotate" else err
        rows.append(r)
    rows.sort(key=lambda r: r["sample_index"])
    (leaf / "metrics.json").write_text(json.dumps(rows, indent=1))


async def _generate_rows(
    targets: list[int],
    *,
    sampler: dict,
    log_path: str,
    task_prompt: str,
    system_prompt: str | None,
    n_eval: int,
    pool_size: int,
    n_shot: int,
    shot_pool_size: int,
    main_task: dict,
    side_task: dict,
    eval_max_tokens: int,
    temperature: float,
    seed: int,
    side_input_word_map: dict | None = None,
    side_explanation: str | None = None,
    monitor: dict | None = None,
    log_prompts: bool = False,
    error_position: int | None = None,
    error_value: int | None = None,
    swap_eval: bool = False,
    model_generated_scheme: bool = False,
    scheme_request_template: str | None = None,
    env_kind: str = "reasoning",
    decoder_template: str | None = None,
    min_prefix: bool = False,
):
    # The pool is built and shuffled at fixed size pool_size (part of the config identity),
    # never as a function of n_eval — so sample_index k names the same sample forever and
    # top-ups extend the exact set a larger fresh run would have produced (this is what keeps
    # same-seed cells pairable sample-by-sample across sessions).
    prep_key = json.dumps(
        [main_task, side_task, side_input_word_map, side_explanation, seed, pool_size, n_shot, shot_pool_size, min_prefix],
        sort_keys=True, default=str)

    def _prepare():
        with _PREP_LOCK:
            if prep_key not in _PREP_CACHE:
                main = get_main_task(**main_task)
                side = get_side_task(**side_task)
                dataset = EncodedDataset(main, side, min_prefix=min_prefix)
                explanation = side_explanation if side_explanation else side.explanation
                rng = random.Random(seed)
                pool = remap_side_input(dataset.test(limit=pool_size), side_input_word_map)
                rng.shuffle(pool)
                shots = sample_shots(remap_side_input(dataset.train(limit=shot_pool_size), side_input_word_map), n_shot, rng)
                _PREP_CACHE[prep_key] = (main, explanation, pool, shots)
            return _PREP_CACHE[prep_key]

    main, explanation, pool, shots = await asyncio.to_thread(_prepare)
    if targets and max(targets) >= len(pool):
        raise ValueError(f"sample_index {max(targets)} exceeds the {len(pool)}-sample pool "
                         f"(pool_size={pool_size}; n_eval counts per position under rotate)")
    monitor_cfg = MonitorConfig(**monitor) if monitor else None

    tokenizer = None
    if error_position is not None:
        model_name = sampler.get("base_model") or sampler["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sampler_obj = await make_sampler(sampler)

    common = dict(side_explanation=explanation, system_prompt=system_prompt, shots=shots,
                  model_generated_scheme=model_generated_scheme, scheme_request_template=scheme_request_template)

    def make_env(i):
        sample = pool[i]
        if env_kind == "messaging":
            return MessagingEnv(sample, main, task_prompt=task_prompt, decoder_template=decoder_template, **common)
        pos = error_position
        if pos == "rotate":  # one cell covers every position: sample i is hit at i mod seq_length
            pos = i % len(sample.expected_reasoning)
        value = error_value
        if pos is not None and value is None:
            value = (sample.expected_reasoning[pos] + 1) % len(sample.alphabet)
        swap_sample = pool[(i + 1) % len(pool)] if swap_eval else None
        return ReasoningEnv(sample, main, task_prompt=task_prompt, error_position=pos,
                            error_value=value, tokenizer=tokenizer, swap_sample=swap_sample, **common)

    envs = [make_env(i) for i in targets]

    async def grade(env):
        """Run one sample to its ScoreResult, or None if its requests failed after retries.
        Keeps a single flaky request from sinking the whole cell."""
        try:
            return await run_inference(env, sampler_obj, max_tokens=eval_max_tokens,
                                       temperature=temperature, monitor=monitor_cfg)
        except Exception as e:
            print(f"  dropped sample ({type(e).__name__}: {e})")
            return None

    scored = await asyncio.gather(*[grade(e) for e in envs])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_ok = 0
    with open(Path(log_path) / "completions.jsonl", "a") as f:
        for i, env, r in zip(targets, envs, scored):
            if r is None:
                continue
            entry = {"sample_index": i, "timestamp": stamp, **r.to_log_entry()}
            if env.designed_scheme is not None:
                entry["designed_scheme"] = env.designed_scheme
            if log_prompts:
                entry["prompt"] = env.opening_messages()
            f.write(json.dumps(entry) + "\n")
            n_ok += 1
    if n_ok == 0:
        raise RuntimeError(f"all {len(targets)} samples failed after retries")
    if n_ok < len(targets):
        print(f"WARNING: dropped {len(targets) - n_ok}/{len(targets)} samples that failed after retries")
    print(f"{log_path}: appended {n_ok} datapoints", flush=True)


def main(**kwargs):
    asyncio.run(_generate(**kwargs))
