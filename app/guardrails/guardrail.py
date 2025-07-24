"""
Guardrail Service for validating input data against defined rules.
"""

from dataclasses import dataclass
import re
from app.models import GuardrailResponse

MIN_LENGTH = 5  

@dataclass
class GuardrailService:
    config: dict

    def validate(self, input_data):
        """Validate input data against guardrail rules."""
        for rule in self.config.get("rules", []):
            if rule["type"] == "pattern":
                _, fixed_input = self._validate_pattern(input_data, rule.get("pattern"), rule.get("replace_with"))
                input_data = fixed_input

        for rule in self.config.get("rules", []):
            if rule["type"] == "length":
                is_valid, input_data = self._validate_length(
                    input_data, rule.get("min_length"), rule.get("max_length")
                )
                if not is_valid:
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

    def _validate_pattern(self, input_data, pattern, replace_with=""):
        if not pattern:
            return True, input_data
        cleaned = re.sub(pattern, replace_with, input_data).strip()
        return True, cleaned

    def _validate_length(self, input_data, min_length=None, max_length=None):
        min_length = min_length or MIN_LENGTH
        max_length = max_length or float("inf")
        length = len(input_data)
        if  length < min_length:
            return False, input_data
        if  length > max_length:
            temp = input_data[:max_length]
            return True, temp
        return True, input_data

    # def _validate_llm(self, input_data, llm_model, validation_prompt):
    #     # TODO: Implement LLM validation logic
    #     raise NotImplementedError("LLM validation is not implemented yet.")
