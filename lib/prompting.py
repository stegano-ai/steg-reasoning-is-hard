import random
from typing import Sequence


def remap_side_input(samples, word_map):
    if word_map:
        for s in samples:
            s.side_input = [word_map[v] for v in s.side_input]
    return samples


def sample_shots(items: Sequence, n_shot: int, rng: random.Random) -> list:
    shots = list(items)
    rng.shuffle(shots)
    return shots[:n_shot]


def user_message(sample, task_prompt, scheme, side_explanation):
    """The task prompt for one sample, with the encoding scheme filled in."""
    return task_prompt.format(
        MAIN_TASK=sample.main_prompt,
        MAIN_TASK_ANSWER=sample.main_answer,
        VALUES=sample.expected_reasoning,
        N_VALUES=len(sample.expected_reasoning),
        MAX_VALUE=sample.max_value,
        ENCODING_SCHEME=scheme,
        SIDE_INPUT=sample.side_input,
        SIDE_TASK_EXPLANATION=side_explanation,
    )


def reference_completion(sample):
    """The gold assistant turn for a sample: its encoded reasoning, plus a `#### <answer>`
    line if the side task has an expected output. Used for few-shot shots and SFT targets."""
    body = sample.render(sample.expected_reasoning)
    if sample.expected_answer is None:
        return body
    return f"{body}\n\n#### {' '.join(str(x) for x in sample.expected_answer)}"
