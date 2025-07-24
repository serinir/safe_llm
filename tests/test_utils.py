"""
Tests for utility functions.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open
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

    def test_valid_pattern_rule_with_all_fields(self):
        """Test valid pattern rule with all optional fields."""
        valid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [
                        {
                            "type": "pattern",
                            "pattern": r"\d+",
                            "replace_with": "X",
                            "error_message": "No digits allowed"
                        }
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert isinstance(config, dict)
            assert "guardrails" in config
            assert len(config["guardrails"]) == 1
            assert config["guardrails"][0]["rules"][0]["pattern"] == r"\d+"
        finally:
            os.unlink(temp_file)

    def test_valid_length_rule_with_min_only(self):
        """Test valid length rule with only min_length."""
        valid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [
                        {
                            "type": "length",
                            "min_length": 5
                        }
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert isinstance(config, dict)
            assert config["guardrails"][0]["rules"][0]["min_length"] == 5
        finally:
            os.unlink(temp_file)

    def test_valid_length_rule_with_max_only(self):
        """Test valid length rule with only max_length."""
        valid_config = {
            "guardrails": [
                {
                    "name": "TestGuardrail",
                    "guardrail_type": "input",
                    "rules": [
                        {
                            "type": "length",
                            "max_length": 100
                        }
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert isinstance(config, dict)
            assert config["guardrails"][0]["rules"][0]["max_length"] == 100
        finally:
            os.unlink(temp_file)

    def test_empty_guardrails_list(self):
        """Test config with empty guardrails list."""
        valid_config = {
            "guardrails": [],
            "prediction": {
                "model": "test_model"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert isinstance(config, dict)
            assert "guardrails" in config
            assert len(config["guardrails"]) == 0
        finally:
            os.unlink(temp_file)

    def test_config_without_guardrails_key(self):
        """Test config without guardrails key."""
        valid_config = {
            "prediction": {
                "model": "test_model"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert isinstance(config, dict)
            # Should still load successfully
            assert "prediction" in config
        finally:
            os.unlink(temp_file)

    def test_permission_error_handling(self):
        """Test handling of permission errors when reading config."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                load_config("restricted_file.json")

    def test_io_error_handling(self):
        """Test handling of IO errors when reading config."""
        with patch("builtins.open", side_effect=IOError("Disk error")):
            with pytest.raises(IOError):
                load_config("corrupted_file.json")
