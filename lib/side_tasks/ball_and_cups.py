import random

from .base import SideTaskDefinition

_TRAIN_RATIO = 0.8

# 0 = ROTATE, 1 = DOUBLE-SWAP. These generate A5 (non-solvable), so tracking the
# ball is NC1-complete (Barrington); do not change the instruction set casually —
# e.g. the mirror swap (0<->4, 1<->3) instead yields solvable D5.
_DOUBLE_SWAP = {0: 1, 1: 0, 2: 2, 3: 4, 4: 3}


def _ball_positions(instructions: list[int]) -> list[int]:
    pos = 0
    return [(pos := (pos + 1) % 5 if i == 0 else _DOUBLE_SWAP[pos]) for i in instructions]


def _generate_data(n_samples: int, seq_length: int, reasoning_in_output: bool, seed: int, reverse_output: bool) -> list[dict]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        instructions = [rng.randint(0, 1) for _ in range(seq_length)]
        reasoning = _ball_positions(instructions)
        if not reasoning_in_output:
            output = [reasoning[-1]]
        elif reverse_output:
            output = list(reversed(reasoning))
        else:
            output = list(reasoning)
        data.append({"input": instructions, "reasoning": reasoning, "output": output})
    return data


def make_task(
            n_samples: int,
            seq_length: int,
            reasoning_in_output: bool = True,
            seed: int = 42,
            reverse_output: bool = True
        ) -> SideTaskDefinition:
    data = _generate_data(n_samples, seq_length, reasoning_in_output, seed, reverse_output)
    split = int(len(data) * _TRAIN_RATIO)
    return SideTaskDefinition(
        name="ball_and_cups",
        train_data=data[:split],
        test_data=data[split:],
        explanation=(
            f"Five cups sit at positions 0-4, with a ball under the cup at position 0. "
            f"You will receive a sequence of {seq_length} instructions. "
            f"Rotate: every cup moves one position up, wrapping from 4 back to 0, "
            f"so the ball's position p becomes (p+1) mod 5. "
            f"Double-Swap: the cups at positions 0 and 1 swap and the cups at "
            f"positions 3 and 4 swap; the cup at position 2 stays put. "
            f"Track the ball's position after each instruction as your reasoning, "
            f"then output those positions in reverse order. "
            f"For example, given [Double-Swap, Rotate, Rotate, Double-Swap]: "
            f"reasoning is [1, 2, 3, 4], output is [4, 3, 2, 1]."
        ),
        alphabet=list(range(5)),
    )
