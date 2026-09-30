import asyncio
import json
import random
from pathlib import Path

import tinker
from tinker_cookbook import checkpoint_utils, renderers, model_info
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, TrainingClientEvaluator
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

from lib import cost_estimate
from lib import inkling_renderer  # noqa: F401  registers tml_v0_no_thinking
from lib.encoded_dataset import EncodedDataset
from lib.prompting import user_message
from lib.main_tasks import get_main_task
from lib.sampling import parse_response_text
from lib.scoring import mean_metrics, score_local
from lib.side_tasks import get_side_task

RECIPE_NAME = "steganography_benchmark_sft"


class ThresholdReached(Exception):
    def __init__(self, state_path: str, metrics: dict[str, float], step: int):
        self.state_path = state_path
        self.metrics = metrics
        self.step = step
        super().__init__(f"early-stop threshold reached: {metrics}")


class SFTDataset(SupervisedDataset):
    def __init__(self, data: list[tinker.Datum], batch_size: int, seed: int = 0):
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        random.Random(seed).shuffle(self.data)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        return self.data[start:min(start + self.batch_size, len(self.data))]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed: int = 0):
        random.Random(self.seed + seed).shuffle(self.data)


def build_datasets(renderer, main, side, *, batch_size, task_prompt, n_train,
                   n_test, n_eval, max_tokens, min_prefix, closed_prefix_targets,
                   seed, log_path):
    tokenizer = renderer.tokenizer
    dataset = EncodedDataset(main, side, min_prefix=min_prefix)

    def user_text(s):
        return user_message(s, task_prompt, main.description, side.explanation)

    def datum(s, assistant):
        from tinker_cookbook.renderers.base import TrainOnWhat
        return conversation_to_datum(
            [{"role": "user", "content": user_text(s)},
             {"role": "assistant", "content": assistant}],
            renderer, max_length=None,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )

    def prefill_datum(s, prefill):
        import torch
        from tinker_cookbook.supervised.common import datum_from_model_input_weights
        messages = [{"role": "user", "content": user_text(s)}]
        base = renderer.build_generation_prompt(messages)
        full = renderer.build_generation_prompt(messages, prefill=prefill)
        weights = torch.zeros(full.length)
        weights[base.length:] = 1.0
        return datum_from_model_input_weights(full, weights, max_length=None, reduction="mean")

    # closed_prefix_targets trains truncated covers as complete (closed) assistant
    # messages instead of unclosed prefills — TML models' turn structure degrades under
    # thousands of steps of never-closed targets (leaked headers, unstoppable loops).
    make_datum = prefill_datum if min_prefix and not closed_prefix_targets else datum

    def build(s):
        rendered = s.render(s.expected_reasoning)
        if rendered is None:  # min-prefix search found no prefix that carries the values
            return None
        n = (len(tokenizer.encode(user_text(s), add_special_tokens=False))
             + len(tokenizer.encode(rendered, add_special_tokens=False)))
        return make_datum(s, rendered) if n <= max_tokens else None

    # Scan lazily in a seeded-shuffled order and stop once we have enough: the kept set
    # is still a uniform random subset, but the (tokenize + min-prefix) work is only paid
    # for samples we keep, not the whole pool. This is what makes the -p cost gate fast.
    samples = dataset.train()
    random.Random(seed).shuffle(samples)
    n_needed = n_eval + n_test + n_train
    kept, scanned = [], 0
    for s in samples:
        scanned += 1
        d = build(s)
        if d is not None:
            kept.append((s, d))
            if len(kept) == n_needed:
                break
    print(f"Roundtrip + length filter: kept {len(kept)} of {scanned} scanned "
          f"(pool size {len(samples)})")
    if len(kept) < n_needed:
        raise ValueError(f"not enough samples passed the roundtrip+length filter: "
                         f"required={n_needed}, available={len(kept)}; raise max_tokens, "
                         f"set min_prefix, or lower n_train/n_test/n_eval")

    n_reserved = n_eval + n_test
    eval_pool = [s for s, _ in kept[:n_eval]]
    test_pool = [s for s, _ in kept[n_eval:n_reserved]]
    test_data = [d for _, d in kept[n_eval:n_reserved]]
    train_data = [d for _, d in kept[n_reserved:]]
    print(f"Train: {len(train_data)}, Test: {len(test_pool)}, Eval: {len(eval_pool)}")

    # NLL baselines on the held-out test set, all over the same (truncated) cover:
    # shuffled = gold values permuted; original = the bare cover (encode nothing).
    rng_shuffle = random.Random(99)
    shuffled_datums = []
    for s in test_pool:
        sv = s.expected_reasoning[:]
        rng_shuffle.shuffle(sv)
        shuffled_datums.append(make_datum(s, s.render(sv)))
    original_datums = [make_datum(s, s.render([])) for s in test_pool]

    nll_evaluators = [NLLEvaluator(shuffled_datums, name="test/shuffled"),
                      NLLEvaluator(original_datums, name="test/original")]

    eval_prefill = sum(len(tokenizer.encode(user_text(s), add_special_tokens=False))
                       for s in eval_pool)
    token_stats = {
        "train_datums": len(train_data),
        "train_tokens_total": sum(d.model_input.length for d in train_data),
        "train_tokens_mean": sum(d.model_input.length for d in train_data) / len(train_data),
        "nll_test_tokens": sum(d.model_input.length for d in test_data),
        "nll_shuffled_tokens": sum(d.model_input.length for d in shuffled_datums),
        "nll_original_tokens": sum(d.model_input.length for d in original_datums),
        "eval_prefill_tokens": eval_prefill,
    }
    print("TOKEN_STATS " + json.dumps(token_stats))
    (Path(log_path) / "token_stats.json").write_text(json.dumps(token_stats))

    return (SFTDataset(train_data, batch_size, seed),
            SFTDataset(test_data, batch_size, seed),
            eval_pool, nll_evaluators)


class GenerationEvaluator(SamplingClientEvaluator):
    def __init__(self, eval_items, task, side_explanation, renderer, task_prompt, eval_max_tokens, log_path):
        self.eval_items = eval_items
        self.task = task
        self.side_explanation = side_explanation
        self.renderer = renderer
        self.task_prompt = task_prompt
        self.log_path = log_path
        self.sampling_params = tinker.SamplingParams(
            max_tokens=eval_max_tokens, temperature=0.0, stop=renderer.get_stop_sequences(),
        )
        self._step_counter = 0

    async def _eval_single(self, sampling_client, item, i):
        messages = [{"role": "user", "content": user_message(
            item, self.task_prompt, self.task.description, self.side_explanation)}]
        prompt = self.renderer.build_generation_prompt(messages)
        resp = await sampling_client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=self.sampling_params,
        )
        try:
            content = parse_response_text(self.renderer, resp.sequences[0].tokens)
        except RendererError:
            # A completion containing stray stop/structure tokens is unparseable; score it
            # as empty rather than killing the whole run (observed: nemotron3 + whitespace-5
            # full-cover eval at step 0, deterministic under temperature 0).
            content = ""
        result = score_local(
            completion=content,
            main_task=self.task,
            expected_reasoning=item.expected_reasoning,
            expected_output=item.expected_answer,
        )
        print(
            f"[eval {i+1}/{len(self.eval_items)}] "
            f"reasoning_correct={result.reasoning_correct:.0%} "
            f"reasoning_exact={result.reasoning_exact:.0f} "
            f"output_exact={result.output_exact:.0f}"
        )
        return result

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        results = await asyncio.gather(*[
            self._eval_single(sampling_client, item, i)
            for i, item in enumerate(self.eval_items)
        ])
        metrics = mean_metrics(results)

        eval_path = Path(self.log_path) / "eval_completions.jsonl"
        with open(eval_path, "a") as f:
            for r in results:
                f.write(json.dumps({"step": self._step_counter, **r.to_log_entry()}) + "\n")
        self._step_counter += 1

        print(f"Eval metrics: {metrics}")
        return metrics


class EarlyStopEvaluator(TrainingClientEvaluator):
    """Stop training once `metric` clears `threshold`, but never before `min_steps` training
    steps have run. The trainer hands us no step number, so steps are derived: this evaluator
    fires every `eval_every` steps, so the k-th call sits at step k * eval_every."""

    def __init__(self, generation_evaluator: GenerationEvaluator, metric: str,
                 threshold: float, min_steps: int, eval_every: int, log_path: str):
        self.generation_evaluator = generation_evaluator
        self.metric = metric
        self.threshold = threshold
        self.min_steps = min_steps
        self.eval_every = eval_every
        self.log_path = log_path
        self._n_evals = 0

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        metrics = await self.generation_evaluator(sampling_client)
        value = metrics[self.metric]
        steps_done = self._n_evals * self.eval_every
        print(f"Early-stop check: {self.metric}={value:.4f} threshold={self.threshold} "
              f"step={steps_done} min_steps={self.min_steps}")
        if steps_done >= self.min_steps and value >= self.threshold:
            print("Threshold reached, stopping")
            paths = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"early_stop_{self._n_evals}",
                log_path=self.log_path,
                loop_state={"batch": self._n_evals},
                kind="state",
            )
            raise ThresholdReached(paths["state_path"], metrics, steps_done)
        self._n_evals += 1
        return metrics


def append_metrics_row(log_path: str, metrics: dict[str, float], step: int | None) -> None:
    """Append an eval row that train.main never got to write: early stopping raises out of
    the evaluator before the row is built, and the post-training eval runs after the loop
    has exited. `step` of None means "just after the last row the trainer wrote"."""
    path = Path(log_path) / "metrics.jsonl"
    if step is None:
        with path.open() as f:
            last = f.readlines()[-1]
        step = json.loads(last)["step"] + 1
    with path.open("a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")


async def train_sft(
    model_name: str,
    log_path: str,
    sft: dict,
    task_prompt: str,
    n_train: int,
    n_test: int,
    n_eval: int,
    batch_size: int,
    main_task: dict,
    side_task: dict,
    max_tokens: int,
    min_prefix: bool,
    closed_prefix_targets: bool,
    renderer_name: str | None,
    wandb_project: str | None,
    wandb_name: str | None,
    eval_max_tokens: int,
    seed: int,
    infrequent_eval_every: int | None,
    early_stop_metric: str | None,
    early_stop_threshold: float | None,
    early_stop_min_steps: int,
):
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    print(f"Using renderer: {renderer_name}")
    Path(log_path).mkdir(parents=True, exist_ok=True)

    task = get_main_task(**main_task)
    side = get_side_task(**side_task)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    train_ds, test_ds, eval_items, nll_evaluators = build_datasets(
        renderer, task, side, batch_size=batch_size, task_prompt=task_prompt,
        n_train=n_train, n_test=n_test, n_eval=n_eval,
        max_tokens=max_tokens, min_prefix=min_prefix,
        closed_prefix_targets=closed_prefix_targets, seed=seed, log_path=log_path,
    )

    generation_evaluator = None
    if n_eval > 0:
        generation_evaluator = GenerationEvaluator(
            eval_items=eval_items,
            task=task,
            side_explanation=side.explanation,
            renderer=renderer,
            task_prompt=task_prompt,
            eval_max_tokens=eval_max_tokens,
            log_path=log_path,
        )

    if early_stop_metric is not None:
        if generation_evaluator is None:
            raise ValueError("early_stop_metric requires n_eval > 0")
        if early_stop_threshold is None:
            raise ValueError("early_stop_threshold required when early_stop_metric is set")
        if infrequent_eval_every is None:
            raise ValueError("early_stop_metric requires infrequent_eval_every")
        infrequent_evaluator = EarlyStopEvaluator(
            generation_evaluator, early_stop_metric, early_stop_threshold,
            early_stop_min_steps, infrequent_eval_every, log_path)
    else:
        infrequent_evaluator = generation_evaluator

    infrequent_kwargs = {}
    if infrequent_evaluator is not None and infrequent_eval_every is not None:
        infrequent_kwargs = {
            "infrequent_evaluator_builders": [lambda: infrequent_evaluator],
            "infrequent_eval_every": infrequent_eval_every,
        }

    config = train.Config(
        **sft,
        model_name=model_name,
        recipe_name=RECIPE_NAME,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=lambda: (train_ds, test_ds),
        evaluator_builders=[lambda e=e: e for e in nll_evaluators],
        **infrequent_kwargs,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    # Informational: the ceiling this run can bill, from the counts just measured. It
    # needs the live price feed, and a run must not die over a docs website — degrade loudly.
    try:
        stage_cost = cost_estimate.max_stage_cost(
            stats=json.loads((Path(log_path) / "token_stats.json").read_text()),
            model_name=model_name, sft=sft, batch_size=batch_size,
            n_eval=len(generation_evaluator.eval_items) if generation_evaluator else 0,
            eval_max_tokens=(generation_evaluator.sampling_params.max_tokens
                             if generation_evaluator else 0),
            infrequent_eval_every=infrequent_eval_every,
        )
        print(cost_estimate.format_stage_cost(log_path, stage_cost))
    except Exception as e:
        print(f"COST ceiling unavailable ({type(e).__name__}: {e}); training continues")

    try:
        await train.main(config)
    except ThresholdReached as e:
        print(f"Early-stop threshold reached ({early_stop_metric}="
              f"{e.metrics[early_stop_metric]:.4f} >= {early_stop_threshold}); ending run")
        append_metrics_row(log_path, e.metrics, e.step)
        return

    if generation_evaluator:
        final_ckpt = checkpoint_utils.get_last_checkpoint(log_path)
        service_client = tinker.ServiceClient()
        sampling_client = await service_client.create_sampling_client_async(model_path=final_ckpt.sampler_path)
        append_metrics_row(log_path, await generation_evaluator(sampling_client), None)


def main(**kwargs):
    asyncio.run(train_sft(**kwargs))
