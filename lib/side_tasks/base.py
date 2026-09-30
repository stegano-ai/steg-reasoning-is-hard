from dataclasses import dataclass


@dataclass
class SideTaskDefinition:
    name: str
    train_data: list[dict]  # each: {"input": ..., "reasoning": list[int], "output": list[int]}
    test_data: list[dict]
    explanation: str  # prompt fragment explaining the side task to the model
    alphabet: list[int]  # the values a single reasoning position can take
