"""
Guardrail Service for validating input data against defined rules.
"""

from dataclasses import dataclass
import re
from app.models import GuardrailResponse


@dataclass
class GuardrailService:
    config: dict

    def validate(self, input_data):
        """Validate input data against guardrail rules."""
        for rule in self.config.get("rules", []):
            if rule["type"] == "pattern":
                if not self._validate_pattern(input_data, rule.get("pattern")):
                    return GuardrailResponse(
                        is_valid=False,
                        message=rule.get(
                            "error_message",
                            "Input does not match the required pattern.",
                        ),
                        guardrail_used=self.config["name"],
                        failed_rule="pattern",
                    )
            elif rule["type"] == "length":
                if not self._validate_length(
                    input_data, rule.get("min_length"), rule.get("max_length")
                ):
                    return GuardrailResponse(
                        is_valid=False,
                        message="Input length is invalid.",
                        guardrail_used=self.config["name"],
                        failed_rule="length",
                    )
            # elif rule['type'] == 'llm':
            #     if not self._validate_llm(input_data, rule.get('llm_model'), rule.get('validation_prompt')):
            #         return False
        return GuardrailResponse(
            is_valid=True, message="Input is valid.", guardrail_used=self.config["name"]
        )

    def _validate_pattern(self, input_data, pattern):
        if not pattern:
            return True
        return not bool(len(re.findall(pattern, input_data)) > 0)

    def _validate_length(self, input_data, min_length=None, max_length=None):
        length = len(input_data)
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True

    # def _validate_llm(self, input_data, llm_model, validation_prompt):
    #     # TODO: Implement LLM validation logic
    #     raise NotImplementedError("LLM validation is not implemented yet.")
