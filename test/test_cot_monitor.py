import json
import pytest
from lib.cot_monitor import _extract_json

pytestmark = pytest.mark.slow

@pytest.mark.parametrize("input_str,expected_suspiciousness", [
    # Normal cases
    ('{"suspiciousness": 7}', 7),
    ('{"reasoning": "looks fine", "suspiciousness": 3}', 3),
    ('{"suspiciousness": 10, "reasoning": "clearly bad"}', 10),

    # Markdown code blocks
    ('```json\n{"suspiciousness": 5}\n```', 5),
    ('```\n{"suspiciousness": 8}\n```', 8),

    # Preamble text
    ('Here is my assessment:\n{"suspiciousness": 4}', 4),
    ('The reasoning appears suspicious.\n\n{"reasoning": "test", "suspiciousness": 9}', 9),

    # Extra whitespace
    ('  {"suspiciousness": 2}  ', 2),
    ('\n\n{"suspiciousness": 6}\n\n', 6),

    # Extra keys
    ('{"suspiciousness": 5, "confidence": 0.9, "notes": "blah"}', 5),
    ('{"analysis": "long text", "suspiciousness": 1, "reasoning": "ok"}', 1),

    # Different key ordering
    ('{"reasoning": "first", "suspiciousness": 7, "extra": "last"}', 7),
    ('{"a": 1, "suspiciousness": 3, "z": 2}', 3),

    # Trailing content after JSON
    ('{"suspiciousness": 4}\n\nLet me know if you need more details.', 4),

    # Suspiciousness as string instead of int
    ('{"suspiciousness": "7"}', "7"),
    ('{"reasoning": "ok", "suspiciousness": "3"}', "3"),
])
def test_extract_json_valid(input_str, expected_suspiciousness):
    extracted = _extract_json(input_str)
    parsed = json.loads(extracted)
    assert parsed["suspiciousness"] == expected_suspiciousness


@pytest.mark.parametrize("input_str", [
    'not json at all',
    '{"suspiciousness": }',
    '{suspiciousness: 5}',
    '',
    'The answer is 5',
])
def test_extract_json_malformed_no_crash(input_str):
    result = _extract_json(input_str)
    assert isinstance(result, str)


@pytest.mark.parametrize("input_str,expected_int", [
    ('{"suspiciousness": 7}', 7),
    ('{"suspiciousness": "5"}', 5),
    ('{"suspiciousness": "10"}', 10),
])
def test_suspiciousness_converts_to_int(input_str, expected_int):
    """Test that suspiciousness can be converted to int (as done in train_grpo.py)."""
    extracted = _extract_json(input_str)
    parsed = json.loads(extracted)
    assert int(parsed["suspiciousness"]) == expected_int


@pytest.mark.parametrize("bad_value", [
    "one",
    "high",
    "very suspicious",
    "N/A",
    None,
    [],
    {},
])
def test_invalid_suspiciousness_fallback(bad_value, capsys):
    """Test that invalid suspiciousness values fall back to 5 with warning."""
    from unittest.mock import AsyncMock, patch
    import asyncio
    from lib import cot_monitor

    mock_response = AsyncMock()
    mock_response.usage = None
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = json.dumps({"suspiciousness": bad_value})

    with patch.object(cot_monitor, '_get_client') as mock_get:
        mock_get.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        result = asyncio.run(cot_monitor._monitor_cot_async("task", "output", "model", "{task}{output}"))

    assert result["suspiciousness"] == 5
    captured = capsys.readouterr()
    assert "[cot_monitor]" in captured.out
