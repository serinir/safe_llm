import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHelper:
    def __init__(self, model_name: str = "default_model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model_name = model_name
        if model_name == "default_model":
            self.model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto"
            ).to(self.device)
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from the model based on the provided prompt."""
        generation_kwargs.setdefault("max_length", 200)
        generation_kwargs.setdefault("do_sample", True)
        generation_kwargs.setdefault("temperature", 0.8)

        messages = [
            {
                "role": "system",
                "content": "Your task is to help user create SQL queries, do only that.",
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
        outputs = self.model.generate(**chat_input, max_new_tokens=150).to(self.device)
        decoded_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_message
