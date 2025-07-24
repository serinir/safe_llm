"""
Configuration and fixtures for pytest.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


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
                        "min_length": 5,
                        "max_length": 50,
                    },
                    {
                        "type": "pattern",
                        "pattern": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                        "replace_with": "",
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
                        "min_length": 1,
                        "max_length": 100,
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


# Mock the LLMHelper to avoid loading actual model during tests
@pytest.fixture(autouse=True)
def mock_llm_helper():
    """Mock LLMHelper to avoid loading actual models during tests."""
    with patch("app.routes.LLMHelper") as mock:
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Mocked generated text response"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(temp_config_file):
    """FastAPI test client with mocked config."""
    # Patch both the config loading and force re-initialization of guardrails
    with patch("app.routes.load_config") as mock_load_config:
        with open(temp_config_file, "r") as f:
            config = json.load(f)
        mock_load_config.return_value = config

        # Import the routes module to get access to the guardrail variables
        from app import routes
        from app.guardrails.guardrail import GuardrailService
        
        # Manually reinitialize the guardrails based on the test config
        routes.input_guardrail = None
        routes.output_guardrail = None
        
        for guardrail in config.get("guardrails", []):
            if guardrail["guardrail_type"] == "input":
                routes.input_guardrail = GuardrailService(guardrail)
            elif guardrail["guardrail_type"] == "output":
                routes.output_guardrail = GuardrailService(guardrail)

        # Import after patching and re-initialization
        from main import app
        return TestClient(app)


@pytest.fixture
def config():
    """Alias for test_config for backwards compatibility."""
    return {
        "guardrails": [
            {
                "name": "TestGuardrail",
                "guardrail_type": "input",
                "description": "Test guardrail for unit tests",
                "rules": [
                    {
                        "type": "length",
                        "min_length": 5,
                        "max_length": 50,
                    },
                    {
                        "type": "pattern",
                        "pattern": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                        "replace_with": "",
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
                        "min_length": 1,
                        "max_length": 100,
                    }
                ],
            },
        ],
        "prediction": {
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "cache_dir": "./.cache/",
        },
    }
