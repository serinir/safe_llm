"""
LLM Helper Module
"""

import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPATIBLE_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]
class LLMHelper:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        os.environ["HF_HOME"] = config.get("cache_dir", "./.cache/")
        logger.info(f"Using cache directory: {os.environ['HF_HOME']}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model_name = config.get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
        if self.model_name not in COMPATIBLE_MODELS:
            raise ValueError(
                f"Model {self.model_name} is not compatible. Supported models: {COMPATIBLE_MODELS}"
            )
        logger.info(f"Loading model: {self.model_name}")
        self.temperature = config.get("parameters", {}).get("temperature", 1.0)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto"
            ).to(self.device)
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from the model based on the provided prompt."""
        generation_kwargs.setdefault("max_length", 200)
        generation_kwargs.setdefault("do_sample", True)
        generation_kwargs.setdefault("temperature", 0.8)

        messages = [
            {
                "role": "system",
                "content": "Your task is to help user create SQL queries, do not deviate from that.",
            },
            {"role": "user", "content": prompt},
        ]
        chat_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(self.device)
        outputs = self.model.generate(
            **chat_input, max_new_tokens=150, temperature=self.temperature
        ).to(self.device)
        decoded_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_message
