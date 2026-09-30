from lib.encoded_dataset import Sample


def prefill_committing_value(sample: Sample, position: int, value: int, tokenizer, decode) -> str:
    """The sample's reasoning with `value` placed at `position`, truncated to the shortest
    token prefix whose decode already yields `value` at `position` — commitment at decode
    level, so no continuation can revert or drop the injected unit. Returns the prefill
    as text.

    Passing the correct value gives the correct-continuation prefill; passing a wrong
    value gives the error-injection prefill the model continues from.
    """
    numbers = list(sample.expected_reasoning)
    numbers[position] = value
    true_ids = tokenizer.encode(sample.render(numbers))
    for cut in range(1, len(true_ids) + 1):
        text = tokenizer.decode(true_ids[:cut])
        decoded = decode(text)
        if len(decoded) > position and decoded[position] == value:
            return text
    raise ValueError(f"decode of the full render never yields {value} at position {position}")
