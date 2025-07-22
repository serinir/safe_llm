import json
from logging import config

GUARDRAIL_RULES = {'pattern','length','llm'}
def load_config(file_path):
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration data.
    """

    with open(file_path, 'r') as file:
        config = json.load(file)
    # validate the config structure if necessary
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a JSON object.")
    if 'guardrails' in config:
            for guardrail in config['guardrails']:
                if 'name' not in guardrail or 'guardrail_type' not in guardrail:
                    raise ValueError("Each guardrail must have a 'name' and 'guardrail_type'.")
                if 'rules' not in guardrail or not isinstance(guardrail['rules'], list):
                    raise ValueError("Each guardrail must have a 'rules' list.")
                ### checking rules structure
                for rule in guardrail['rules']:
                    if 'type' not in rule or rule['type'] not in GUARDRAIL_RULES:
                        raise ValueError(f"Invalid rule type: {rule.get('type', 'None')}. Must be one of {GUARDRAIL_RULES}.")
                    if rule['type'] == 'pattern' and 'pattern' not in rule:
                        raise ValueError("Pattern rules must have a 'pattern' key.")
                    if rule['type'] == 'length' and ('min_length' not in rule and 'max_length' not in rule):
                        raise ValueError("Length rules must have 'min_length' or 'max_length' keys.")

    return config