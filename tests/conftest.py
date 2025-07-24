"""
Configuration and fixtures for pytest.
"""

import pytest
import tempfile
import json
import os


@pytest.fixture
def test_config():
    """Test configuration data."""
    return {
        "guardrails": [
            {
                "name": "TestGuardrail",
                "guardrail_type": "input",
                "description": "Test guardrail for unit tests",
                "rules": [
                    {
                        "type": "length",
                        "max_length": 50,
                        "error_message": "Input too long for test",
                    },
                    {
                        "type": "pattern",
                        "pattern": "([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})",
                        "error_message": "Invalid characters in test input",
                    },
                ],
            },
            {
                "name": "TestOutputGuardrail",
                "guardrail_type": "output",
                "description": "Test output guardrail",
                "rules": [
                    {
                        "type": "length",
                        "max_length": 100,
                        "error_message": "Output too long for test",
                    }
                ],
            },
        ],
        "prediction": {
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "cache_dir": "./.cache/",
        },
    }


@pytest.fixture
def temp_config_file(test_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_config, f)
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return {
        "valid_short": "Hello world",
        "valid_long": "This is a longer text that should pass validation",
        "invalid_chars": "how to filter data on jacob@gmail.com!",
        "too_long": "a" * 100,
        "empty": "",
        "similar_text1": "The quick brown fox jumps over the lazy dog",
        "similar_text2": "A quick brown fox leaps over a lazy dog",
        "different_text": "Python is a great programming language",
    }
