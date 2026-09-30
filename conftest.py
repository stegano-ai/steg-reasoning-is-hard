import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked as slow (network calls / HuggingFace dataset loads)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that hit the network (OpenRouter API, HuggingFace dataset loads). Skipped by default; pass --run-slow to enable.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
