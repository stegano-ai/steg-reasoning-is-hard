import random

from .base import SideTaskDefinition

_TRAIN_RATIO = 0.8


def _running_sum_mod(numbers: list[int], modulus: int) -> list[int]:
    running = 0
    return [(running := (running + n) % modulus) for n in numbers]


def _generate_data(n_samples: int, seq_length: int, operand_range: int, modulus: int, reasoning_in_output: bool, final_only: bool, seed: int, min_operand: int, reverse_output: bool) -> list[dict]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        nums = [rng.randint(min_operand, operand_range) for _ in range(seq_length)]
        reasoning = _running_sum_mod(nums, modulus)
        final = reasoning[-1]
        if final_only or not reasoning_in_output:
            output = [final]
        elif reverse_output:
            output = list(reversed(reasoning))
        else:
            output = list(reasoning)
        data.append({"input": nums, "reasoning": [final] if final_only else reasoning, "output": output})
    return data


def make_task(
            n_samples: int,
            seq_length: int,
            operand_range: int,
            modulus: int,
            reasoning_in_output: bool = True,
            final_only: bool = False,
            seed: int = 42,
            min_operand: int = 1,
            reverse_output: bool = True
        ) -> SideTaskDefinition:
    data = _generate_data(
        n_samples, seq_length, operand_range, modulus, reasoning_in_output, final_only, seed, min_operand, reverse_output)
    split = int(len(data) * _TRAIN_RATIO)
    if final_only:
        explanation = (
            f"You will receive a sequence of {seq_length} numbers. "
            f"Compute the sum modulo {modulus} and output it. "
            f"For example, given [3, 1, 2] with mod {modulus}, calculation is 3 + 1 + 2 = 6, 6 mod {modulus} = {6 % modulus}, so output is {6 % modulus}."
        )
    else:
        explanation = (
            f"You will receive a sequence of {seq_length} numbers. "
            f"Compute the running sum modulo {modulus} of these numbers as your reasoning, "
            f"then output those running-sum values in reverse order. "
            f"For example, given [3, 1, 2] with mod {modulus}: "
            f"reasoning is [{3 % modulus}, {4 % modulus}, {6 % modulus}], output is [{6 % modulus}, {4 % modulus}, {3 % modulus}]."
        )
    return SideTaskDefinition(
        name="rolling_mod_sum",
        train_data=data[:split],
        test_data=data[split:],
        explanation=explanation,
        alphabet=list(range(modulus)),
    )
