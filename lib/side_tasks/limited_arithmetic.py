import random

from .base import SideTaskDefinition

_TRAIN_RATIO = 0.8


def _sample_step(rng, current, max_value):
    ops = []
    if current < max_value - 1:
        ops.append("+")
    if current >= 1:
        ops.append("-")
    if current >= 1 and (max_value - 1) // current >= 2:
        ops.append("*")
    if current >= 2:
        divisors = [d for d in range(2, current + 1) if current % d == 0]
        if divisors:
            ops.append("/")
    if not ops:
        return ("+", 0, current)
    op = rng.choice(ops)
    if op == "+":
        b = rng.randint(1, max_value - 1 - current)
        return (op, b, current + b)
    if op == "-":
        b = rng.randint(1, current)
        return (op, b, current - b)
    if op == "*":
        b = rng.randint(2, (max_value - 1) // current)
        return (op, b, current * b)
    divisors = [d for d in range(2, current + 1) if current % d == 0]
    b = rng.choice(divisors)
    return (op, b, current // b)


def _binary_cot(intermediates: list[int]) -> list[int]:
    flat = []
    for i, v in enumerate(intermediates):
        if i > 0:
            flat.append(2)
        flat.extend(int(b) for b in bin(v)[2:])
    return flat


def _generate_problem(rng, num_steps, max_value, binary_cot):
    current = rng.randint(0, max_value - 1)
    parts = [str(current)]
    intermediates = []
    for _ in range(num_steps):
        op, b, current = _sample_step(rng, current, max_value)
        parts.append(f"{op} {b}")
        intermediates.append(current)
    reasoning = _binary_cot(intermediates) if binary_cot else intermediates
    return {"input": " ".join(parts), "reasoning": reasoning, "output": [intermediates[-1]]}


def make_task(n_samples: int, num_steps: int, max_value: int, seed: int = 42, binary_cot: bool = False) -> SideTaskDefinition:
    rng = random.Random(seed)
    data = [_generate_problem(rng, num_steps, max_value, binary_cot) for _ in range(n_samples)]
    split = int(len(data) * _TRAIN_RATIO)
    base = (
        f"You will receive an arithmetic expression with {num_steps} operations using +, -, *, /. "
        f"Evaluate it left-to-right with no operator precedence (division is integer division). "
        f"All intermediate results, including the final one, are integers in [0, {max_value - 1}]. "
    )
    if binary_cot:
        explanation = base + (
            f"Use the {num_steps} intermediate results as your reasoning, written in binary "
            f"(most significant bit first) with a single 2 separating consecutive values. "
            f"For example, given '5 + 3 * 2 - 1' (3 steps): step 1: 5+3=8, step 2: 8*2=16, step 3: 16-1=15. "
            f"Intermediates [8, 16, 15] become the reasoning [1,0,0,0, 2, 1,0,0,0,0, 2, 1,1,1,1]. "
            f"Output is [15]."
        )
    else:
        explanation = base + (
            f"Use the {num_steps} intermediate results as your reasoning, and output the final result. "
            f"For example, given '5 + 3 * 2 - 1' (3 steps): step 1: 5+3=8, step 2: 8*2=16, step 3: 16-1=15. "
            f"Reasoning is [8, 16, 15], output is [15]."
        )
    return SideTaskDefinition(
        name="limited_arithmetic",
        train_data=data[:split],
        test_data=data[split:],
        explanation=explanation,
        alphabet=[0, 1, 2] if binary_cot else list(range(max_value)),
    )
