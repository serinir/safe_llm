"""
Tests for utility functions.
"""

import pytest
import json
import tempfile
import os
from app.utils import load_config


class TestLoadConfig:
    """Test the load_config utility function."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert "guardrails" in config
        assert len(config["guardrails"]) == 2

    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.json")

    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_non_dict_config(self):
        """Test loading a JSON file that doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["this", "is", "a", "list"], f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Configuration file must contain a JSON object"
            ):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_invalid_guardrail_structure(self):
        """Test loading config with invalid guardrail structure."""
        invalid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail"
                    # Missing required fields
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError,
                match="Each guardrail must have a 'name' and 'guardrail_type'",
            ):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_invalid_rules_structure(self):
        """Test loading config with invalid rules structure."""
        invalid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": "not a list",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Each guardrail must have a 'rules' list"
            ):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_invalid_rule_type(self):
        """Test loading config with invalid rule type."""
        invalid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [{"type": "invalid_type"}],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid rule type"):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_pattern_rule_validation(self):
        """Test validation of pattern rules."""
        invalid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [
                        {
                            "type": "pattern"
                            # Missing required 'pattern' field
                        }
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Pattern rules must have a 'pattern' key"
            ):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_length_rule_validation(self):
        """Test validation of length rules."""
        invalid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [
                        {
                            "type": "length"
                            # Missing min_length and max_length
                        }
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError,
                match="Length rules must have 'min_length' or 'max_length' keys",
            ):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)
