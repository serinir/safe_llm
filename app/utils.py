"""
utils file for the Safe LLM Endpoint API
"""

import json
import os

GUARDRAIL_RULES = {
            "pattern",
            "length"
            # "llm"
                   }


def load_config(file_path):
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration data.
    """

    with open(file_path, "r") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a JSON object.")
    

    # Validate guardrails  configuration
    if "guardrails" in config:
        for guardrail in config["guardrails"]:
            if "name" not in guardrail or "guardrail_type" not in guardrail:
                raise ValueError(
                    "Each guardrail must have a 'name' and 'guardrail_type'."
                )
            if "rules" not in guardrail or not isinstance(guardrail["rules"], list):
                raise ValueError("Each guardrail must have a 'rules' list.")
            ### checking rules structure
            for rule in guardrail["rules"]:
                if "type" not in rule or rule["type"] not in GUARDRAIL_RULES:
                    raise ValueError(
                        f"Invalid rule type: {rule.get('type', 'None')}. Must be one of {GUARDRAIL_RULES}."
                    )
                if rule["type"] == "pattern" :
                    if "pattern" not in rule:
                        raise ValueError("Pattern rules must have a 'pattern' key.")
                    if "replace_with" not in rule:
                        rule["replace_with"] = ""
                
                if rule["type"] == "length":
                    if ( "min_length" not in rule and "max_length" not in rule):
                        raise ValueError(
                            "Length rules must have 'min_length' or 'max_length' keys."
                        )
                    if "max_length" in rule and not isinstance(rule["max_length"], int):
                        raise ValueError("max_length must be an integer.")
                    if "min_length" in rule and not isinstance(rule["min_length"], int):
                        raise ValueError("min_length must be an integer.")
                    if "max_length" in rule and rule["max_length"] < 50:
                        raise ValueError("max_length must be at least 50 characters.")
    ## Validate prediction configuration                
    if "prediction" in config:
        if "model" not in config["prediction"]:
            raise ValueError("Prediction configuration must have a 'model' key.")
        if "cache_dir" in config["prediction"]:
            config["prediction"]["cache_dir"] = os.path.expanduser(
                config["prediction"]["cache_dir"]
            )
        if "parameters" not in config["prediction"]:
            config["prediction"]["parameters"] = {}
        if "temperature" not in config["prediction"]["parameters"]:
            config["prediction"]["parameters"]["temperature"] = 1.0
    return config
