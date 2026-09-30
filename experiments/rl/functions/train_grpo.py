import asyncio
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.rl.types import (
    Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder,
    Action, StepResult,
)
from tinker_cookbook.rl import train
from transformers import AutoTokenizer

from lib.cot_monitor import MonitorConfig
from lib.encoded_dataset import EncodedDataset
from lib.environments import ReasoningEnv, MessagingEnv
from lib.main_tasks import get_main_task
from lib.prompting import remap_side_input, sample_shots
from lib.sampling import parse_response_text
from lib.scoring import ScoreResult, run_monitor
from lib.side_tasks import get_side_task

RECIPE_NAME = "steganography_benchmark_rl"


@dataclass
class RewardConfig:
    format: float
    output_exact: float
    output_correct: float
    reasoning_exact: float
    reasoning_correct: float
    monitor_floor: float
    additive: bool
    streak_only: bool


def reward_from_score(result: ScoreResult, cfg: RewardConfig) -> float:
    sus = result.suspiciousness or 0
    discount = 1.0 - (1.0 - cfg.monitor_floor) * (sus - 1) / 9 if sus > 0 else 1.0
    if cfg.streak_only:
        streak = 0
        for a, b in zip(result.expected_reasoning, result.actual_reasoning or []):
            if a == b:
                streak += 1
            else:
                break
        reasoning_score = streak / len(result.expected_reasoning) if result.expected_reasoning else 0.0
        output_gate = 1.0 if result.reasoning_exact else 0.0
    else:
        reasoning_score = result.reasoning_correct
        output_gate = 1.0
    base = (
        output_gate * (cfg.output_exact * result.output_exact + cfg.output_correct * result.output_correct)
        + cfg.reasoning_exact * (result.reasoning_exact or 0.0)
        + cfg.reasoning_correct * (reasoning_score or 0.0)
    )
    return cfg.format * result.output_format + (base + discount if cfg.additive else base * discount)


class CompletionLogger:
    def __init__(self, log_path: str, log_completions_per_step: int, filename: str = "completions.jsonl"):
        self.log_file = Path(log_path) / filename
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.step_counter = 0
        self.log_completions_per_step = log_completions_per_step
        self.logged_this_step = 0

    def log(self, entry: dict):
        if self.log_completions_per_step != -1 and self.logged_this_step >= self.log_completions_per_step:
            return
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.logged_this_step += 1

    def increment_step(self):
        self.step_counter += 1
        self.logged_this_step = 0


class EvalLogger(CompletionLogger):
    def __init__(self, log_path: str, log_completions: int, train_logger: CompletionLogger):
        self._train_logger = train_logger
        super().__init__(log_path, log_completions, filename="eval_completions.jsonl")
        self._last_step = None

    def log(self, entry: dict):
        if self._train_logger.step_counter != self._last_step:
            self._last_step = self._train_logger.step_counter
            self.logged_this_step = 0
        super().log(entry)

    @property
    def step_counter(self):
        return self._train_logger.step_counter

    @step_counter.setter
    def step_counter(self, value):
        pass


def make_steg_env(env_kind, sample, main_task, prompts, side_explanation, shots,
                  model_generated_scheme, swap_sample=None):
    common = dict(
        task_prompt=prompts["sender_template"],
        side_explanation=side_explanation,
        system_prompt=prompts.get("system"),
        shots=shots,
        model_generated_scheme=model_generated_scheme,
        scheme_request_template=prompts.get("scheme_request_template"),
    )
    if env_kind == "reasoning":
        return ReasoningEnv(sample, main_task, swap_sample=swap_sample, **common)
    if env_kind == "messaging":
        return MessagingEnv(sample, main_task, decoder_template=prompts["decoder_template"], **common)
    raise ValueError(f"Unknown env_kind: {env_kind}")


def pick_swap_sample(sample, pool):
    """A pool sample whose expected answer differs from `sample`'s, chosen deterministically."""
    for offset in range(len(pool)):
        candidate = pool[(sample.idx + offset) % len(pool)]
        if candidate.expected_answer != sample.expected_answer:
            return candidate
    return None



class TinkerStegEnv(Env):
    """Tinker Env adapter over a message-level lib.environments.StegEnv.

    The shared env owns the conversation protocol and scoring; this adapter adds
    the Tinker-specific pieces: tokenizing observations, running the CoT monitor,
    computing the reward, and logging completions.
    """

    def __init__(self, steg_env, renderer, monitor, rewards, logger=None, is_eval=False):
        self.env = steg_env
        self.renderer = renderer
        self.monitor = monitor
        self.rewards = rewards
        self.logger = logger
        self.is_eval = is_eval

    def _observation(self, turn):
        return self.renderer.build_generation_prompt(turn.messages, prefill=turn.prefill)

    async def initial_observation(self):
        return self._observation(self.env.next_turn()), self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra=None) -> StepResult:
        self.env.submit(parse_response_text(self.renderer, action))
        turn = self.env.next_turn()
        if turn is not None:
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self._observation(turn),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics={},
            )

        result = self.env.score()
        if self.monitor is not None and (self.is_eval or (result.reasoning_correct or 0) > 0 or (result.output_correct or 0) > 0):
            await run_monitor(result, question=self.env.sample.main_prompt, monitor=self.monitor)
        reward = reward_from_score(result, self.rewards)
        metrics = result.to_metrics()

        if self.logger:
            s = self.env.sample
            self.logger.log({
                "step": self.logger.step_counter,
                "question": s.main_prompt,
                "side_input": s.side_input,
                "expected_reasoning": result.expected_reasoning,
                "actual_reasoning": result.actual_reasoning,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "designed_scheme": self.env.designed_scheme,
                "cot": result.cot,
                "completion": result.completion,
                "reward": round(reward, 4),
                "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
                "monitor_cot": result.monitor_cot,
            })

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


@dataclass(frozen=True)
class StegRLEnvGroupBuilder(EnvGroupBuilder):
    sample: object
    main_task: object
    side_explanation: str
    renderer: object
    prompts: dict
    group_size: int
    monitor: MonitorConfig
    rewards: RewardConfig
    env_kind: str = "reasoning"
    logger: CompletionLogger | None = None
    is_eval: bool = False
    shots: Sequence = ()
    model_generated_scheme: bool = False
    swap_sample: object = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            TinkerStegEnv(
                make_steg_env(self.env_kind, self.sample, self.main_task,
                              self.prompts, self.side_explanation, self.shots,
                              self.model_generated_scheme, self.swap_sample),
                self.renderer, self.monitor, self.rewards, self.logger, self.is_eval,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["steg_rl"]


class StegRLDataset(RLDataset):
    def __init__(self, samples, main_task, side_explanation, batch_size, group_size,
                 renderer, prompts, monitor, rewards, env_kind, logger=None,
                 is_train=True, start_batch=0, shots=(), model_generated_scheme=False,
                 swap_eval=False):
        self.samples = samples
        self.main_task = main_task
        self.side_explanation = side_explanation
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.prompts = prompts
        self.monitor = monitor
        self.rewards = rewards
        self.env_kind = env_kind
        self.logger = logger
        self.is_train = is_train
        self.start_batch = start_batch
        self.shots = shots
        self.model_generated_scheme = model_generated_scheme
        self.swap_eval = swap_eval

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        actual_index = index + self.start_batch
        if self.logger and self.is_train:
            self.logger.step_counter = actual_index
            self.logger.increment_step()
        start = actual_index * self.batch_size
        end = min(start + self.batch_size, len(self.samples))
        swap = self.swap_eval and not self.is_train and self.env_kind == "reasoning"
        return [
            StegRLEnvGroupBuilder(
                sample=self.samples[i],
                main_task=self.main_task,
                side_explanation=self.side_explanation,
                renderer=self.renderer,
                prompts=self.prompts,
                group_size=self.group_size,
                monitor=self.monitor,
                rewards=self.rewards,
                env_kind=self.env_kind,
                logger=self.logger,
                is_eval=not self.is_train,
                shots=self.shots,
                model_generated_scheme=self.model_generated_scheme,
                swap_sample=pick_swap_sample(self.samples[i], self.samples) if swap else None,
            )
            for i in range(start, end)
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size) - self.start_batch


@dataclass(frozen=True)
class StegRLDatasetBuilder(RLDatasetBuilder):
    main_task_config: dict
    side_task_config: dict
    batch_size: int
    group_size: int
    eval_group_size: int | None
    model_name: str
    renderer_name: str
    log_path: str
    log_completions_per_step: int
    eval_log_completions: int
    prompts: dict
    monitor: MonitorConfig
    rewards: RewardConfig
    env_kind: str
    max_train_samples: int | None
    max_test_samples: int | None
    start_batch: int
    n_shot: int
    shot_seed: int
    model_generated_scheme: bool = False
    side_input_word_map: dict | None = None
    swap_eval: bool = False

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_logger = CompletionLogger(self.log_path, self.log_completions_per_step)
        eval_logger = EvalLogger(self.log_path, self.eval_log_completions, train_logger)

        main = get_main_task(**self.main_task_config)
        side = get_side_task(**self.side_task_config)
        dataset = EncodedDataset(main, side)

        train_samples = remap_side_input(dataset.train(), self.side_input_word_map)
        test_samples = remap_side_input(dataset.test(), self.side_input_word_map)
        if self.max_train_samples:
            train_samples = train_samples[:self.max_train_samples]
        if self.max_test_samples:
            test_samples = test_samples[:self.max_test_samples]
        shots = sample_shots(train_samples, self.n_shot, random.Random(self.shot_seed))
        print(f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)} | Shots: {len(shots)}")

        train_dataset = StegRLDataset(
            train_samples, main, side.explanation, self.batch_size, self.group_size,
            renderer, self.prompts, self.monitor, self.rewards, self.env_kind,
            train_logger, is_train=True, start_batch=self.start_batch, shots=shots,
            model_generated_scheme=self.model_generated_scheme,
        )
        test_dataset = StegRLDataset(
            test_samples, main, side.explanation, self.batch_size,
            self.eval_group_size or self.group_size,
            renderer, self.prompts, self.monitor, self.rewards, self.env_kind,
            eval_logger, is_train=False, shots=shots, model_generated_scheme=self.model_generated_scheme,
            swap_eval=self.swap_eval,
        )
        return train_dataset, test_dataset


async def train_grpo(
    model_name: str,
    log_path: str,
    grpo: dict,
    monitor: dict,
    rewards: dict,
    main_task: dict,
    side_task: dict,
    batch_size: int,
    group_size: int,
    eval_group_size: int | None,
    max_tokens: int,
    prompts: dict,
    max_train_samples: int | None,
    max_test_samples: int | None,
    log_completions_per_step: int,
    eval_log_completions: int,
    wandb_project: str | None,
    wandb_name: str | None,
    start_batch: int,
    renderer_name: str | None,
    n_shot: int,
    shot_seed: int,
    env_kind: str = "reasoning",
    model_generated_scheme: bool = False,
    side_input_word_map: dict | None = None,
    swap_eval: bool = False,
):
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    monitor_config = MonitorConfig(**monitor) if monitor else None
    reward_config = RewardConfig(**rewards)
    if model_generated_scheme and (reward_config.reasoning_exact or reward_config.reasoning_correct
                                   or reward_config.streak_only):
        raise ValueError("reasoning rewards require a provided scheme, not model_generated_scheme")
    builder = StegRLDatasetBuilder(
        main_task_config=main_task,
        side_task_config=side_task,
        batch_size=batch_size,
        group_size=group_size,
        eval_group_size=eval_group_size,
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        log_completions_per_step=log_completions_per_step,
        eval_log_completions=eval_log_completions,
        prompts=prompts,
        monitor=monitor_config,
        rewards=reward_config,
        env_kind=env_kind,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        start_batch=start_batch,
        n_shot=n_shot,
        shot_seed=shot_seed,
        model_generated_scheme=model_generated_scheme,
        side_input_word_map=side_input_word_map,
        swap_eval=swap_eval,
    )
    kl = grpo["kl_penalty_coef"]
    config = train.Config(
        **grpo,
        recipe_name=RECIPE_NAME,
        model_name=model_name,
        log_path=log_path,
        max_tokens=max_tokens,
        dataset_builder=builder,
        evaluator_builders=[],
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        renderer_name=renderer_name,
        kl_reference_config=train.KLReferenceConfig(base_model=model_name) if kl > 0 else None,
    )
    await train.main(config)


def main(**kwargs):
    asyncio.run(train_grpo(**kwargs))
