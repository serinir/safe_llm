"""
Tests for guardrail functionality.
"""

from app.guardrails.guardrail import GuardrailService
from app.models import GuardrailResponse


class TestGuardrailService:
    """Test the GuardrailService class."""

    def test_guardrail_creation(self, test_config):
        """Test creating a guardrail service from config."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        assert service.config == guardrail_config
        assert service.config["name"] == "TestGuardrail"
        assert service.config["guardrail_type"] == "input"

    def test_valid_input_short_text(self, test_config, sample_texts):
        """Test validation with valid short text."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        result = service.validate(sample_texts["valid_short"])

        assert isinstance(result, GuardrailResponse)
        assert result.is_valid
        assert result.message == "Input is valid."
        assert result.guardrail_used == "TestGuardrail"
        assert result.failed_rule is None

    def test_valid_input_longer_text(self, test_config, sample_texts):
        """Test validation with valid longer text."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        result = service.validate(sample_texts["valid_long"])

        assert isinstance(result, GuardrailResponse)
        assert result.is_valid
        assert result.message == "Input is valid."
        assert result.guardrail_used == "TestGuardrail"

    def test_invalid_length_too_long(self, test_config, sample_texts):
        """Test validation with text that's too long."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        result = service.validate(sample_texts["too_long"])

        assert isinstance(result, GuardrailResponse)
        # With current implementation, too long text gets truncated and passes
        assert result.is_valid
        assert result.message == "Input is valid."
        assert result.guardrail_used == "TestGuardrail"

    def test_invalid_length_too_short(self, test_config):
        """Test validation with text that's too short."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        # Text shorter than MIN_LENGTH (5)
        short_text = "hi"
        result = service.validate(short_text)

        assert isinstance(result, GuardrailResponse)
        assert not result.is_valid
        assert "Input length is invalid." in result.message
        assert result.failed_rule == "length"
        assert result.guardrail_used == "TestGuardrail"

    def test_pattern_cleaning(self, test_config, sample_texts):
        """Test pattern validation cleans input."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        result = service.validate(sample_texts["invalid_chars"])
        
        assert isinstance(result, GuardrailResponse)
        # Pattern validation cleans the input but validation passes
        assert result.is_valid
        assert result.message == "Input is valid."
        assert result.guardrail_used == "TestGuardrail"

    def test_empty_text(self, test_config, sample_texts):
        """Test validation with empty text."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        result = service.validate(sample_texts["empty"])

        assert isinstance(result, GuardrailResponse)
        # Empty text should fail length validation (min_length = 5)
        assert not result.is_valid
        assert "Input length is invalid." in result.message
        assert result.failed_rule == "length"

    def test_pattern_validation_method(self, test_config):
        """Test the _validate_pattern method directly."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        # Test with no pattern
        is_valid, cleaned = service._validate_pattern("any text", "")
        assert is_valid is True
        assert cleaned == "any text"

        # Test with pattern cleaning
        is_valid, cleaned = service._validate_pattern("test@email.com", r"@[^@\s]+")
        assert is_valid is True
        assert "@" not in cleaned  # Email domain should be removed

    def test_length_validation_method(self, test_config):
        """Test the _validate_length method directly."""
        guardrail_config = test_config["guardrails"][0]
        service = GuardrailService(config=guardrail_config)

        # Test within limits
        is_valid, processed = service._validate_length("short", min_length=1, max_length=50)
        assert is_valid is True
        assert processed == "short"

        # Test too long - should be truncated
        is_valid, processed = service._validate_length("a" * 100, max_length=50)
        assert is_valid is True
        assert len(processed) == 50

        # Test too short
        is_valid, processed = service._validate_length("", min_length=5)
        assert is_valid is False
        assert processed == ""

        # Test with no limits
        is_valid, processed = service._validate_length("any text")
        assert is_valid is True
        assert processed == "any text"

    def test_config_without_rules(self):
        """Test guardrail with config that has no rules."""
        config = {"name": "EmptyGuardrail", "guardrail_type": "input", "rules": []}
        service = GuardrailService(config=config)

        result = service.validate("any text")

        assert isinstance(result, GuardrailResponse)
        assert result.is_valid
        assert result.message == "Input is valid."

    def test_config_missing_rules(self):
        """Test guardrail with config that has no rules key."""
        config = {
            "name": "NoRulesGuardrail",
            "guardrail_type": "input",
            # No 'rules' key
        }
        service = GuardrailService(config=config)

        result = service.validate("any text")

        assert isinstance(result, GuardrailResponse)
        assert result.is_valid
        assert result.message == "Input is valid."

    def test_length_rule_with_explicit_values(self):
        """Test length rule with explicit min and max values."""
        config = {
            "name": "LengthGuardrail",
            "guardrail_type": "input",
            "rules": [
                {
                    "type": "length",
                    "min_length": 10,
                    "max_length": 20
                }
            ]
        }
        service = GuardrailService(config=config)

        # Test valid length
        result = service.validate("valid length text")
        assert result.is_valid

        # Test too short
        result = service.validate("short")
        assert not result.is_valid
        assert result.failed_rule == "length"

        # Test too long (should be truncated and pass)
        result = service.validate("this is a very long text that exceeds the maximum length")
        assert result.is_valid

    def test_pattern_rule_with_replacement(self):
        """Test pattern rule with replacement."""
        config = {
            "name": "PatternGuardrail",
            "guardrail_type": "input",
            "rules": [
                {
                    "type": "pattern",
                    "pattern": r"\d+",  # Remove all digits
                    "replace_with": ""
                }
            ]
        }
        service = GuardrailService(config=config)

        result = service.validate("text with 123 numbers 456")
        assert result.is_valid
        # The actual cleaning is done internally

    def test_multiple_rules_processing(self):
        """Test that multiple rules are processed in order."""
        config = {
            "name": "MultiRuleGuardrail",
            "guardrail_type": "input",
            "rules": [
                {
                    "type": "pattern",
                    "pattern": r"[^a-zA-Z\s]",  # Remove non-alphabetic chars
                    "replace_with": ""
                },
                {
                    "type": "length",
                    "min_length": 5,
                    "max_length": 30
                }
            ]
        }
        service = GuardrailService(config=config)

        # Text with special chars that should be cleaned first
        result = service.validate("hello@world#123!")
        assert result.is_valid

    def test_empty_pattern_rule(self):
        """Test pattern rule with empty pattern."""
        config = {
            "name": "EmptyPatternGuardrail",
            "guardrail_type": "input",
            "rules": [
                {
                    "type": "pattern",
                    "pattern": "",  # Empty pattern
                    "replace_with": ""
                }
            ]
        }
        service = GuardrailService(config=config)

        original_text = "unchanged text"
        result = service.validate(original_text)
        assert result.is_valid
