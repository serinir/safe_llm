from dataclasses import dataclass
from enum import Enum


@dataclass
class Guardrail:
    type: Enum['Input', 'Output']
    config: dict

    def validate(self, input_data):
        if self.type == 'Input':
            return self._validate_input(input_data)
        elif self.type == 'Output':
            return self._validate_output(input_data)
        else:
            raise ValueError(f"Unknown guardrail type: {self.type}")

    def _validate_output(self, output_data):
        pass

    def _validate_input(self, input_data):
        for rule in self.config.get('rules', []):
            if rule['type'] == 'pattern':
                if not self._validate_pattern(input_data, rule.get('pattern', '')):
                    return False
            elif rule['type'] == 'length':
                if not self._validate_length(input_data, rule.get('min_length'), rule.get('max_length')):
                    return False
            elif rule['type'] == 'llm':
                if not self._validate_llm(input_data, rule.get('llm_model', ''), rule.get('validation_prompt', '')):
                    return False
        return True
    
    def _validate_pattern(self, input_data, pattern):
        import re
        if not pattern:
            return True
        return bool(re.match(pattern, input_data))
    
    def _validate_length(self, input_data, min_length=None, max_length=None):
        length = len(input_data)
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True

    def _validate_llm(self, input_data, llm_model, validation_prompt):
        # TODO: Implement LLM validation logic
        raise NotImplementedError("LLM validation is not implemented yet.")
