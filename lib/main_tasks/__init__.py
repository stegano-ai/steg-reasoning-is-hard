from .base import MainTask


def get_main_task(type: str, **kwargs) -> MainTask:
    if type == "aqua_rat":
        from .aqua_rat import AquaRatTask
        return AquaRatTask(**kwargs)
    if type == "wildchat":
        from .wildchat import WildChatTask
        return WildChatTask(**kwargs)
    if type == "dialogue":
        from .dialogue import DialogueTask
        return DialogueTask(**kwargs)
    if type == "knapsack":
        from .knapsack import KnapsackTask
        return KnapsackTask(**kwargs)
    if type == "blank":
        from .blank import BlankTask
        return BlankTask(**kwargs)
    raise ValueError(f"Unknown main task type: {type}")
