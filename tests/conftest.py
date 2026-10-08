"""Pytest configuration for tests."""
import sys
from pathlib import Path

# Add project root to path so 'src' imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os

import pytest

# Dummy keys so clients that are constructed but mocked (no real calls) can initialise.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.setdefault(_k, "test-dummy-key")


def pytest_collection_modifyitems(config, items):
    """Tests marked `network` (HF downloads, real API clients) are skipped unless CORIN_NETWORK_TESTS=1."""
    if os.environ.get("CORIN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set CORIN_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
