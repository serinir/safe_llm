"""
Tests for LLM helper functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from app.llm_helper import LLMHelper


class TestLLMHelper:
    """Test the LLMHelper class."""

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_initialization_cpu(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Test LLMHelper initialization with CPU."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Test initialization with config
        config = {
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "cache_dir": "/tmp/test_cache",
            "parameters": {"temperature": 0.5}
        }
        helper = LLMHelper(config=config)

        assert helper.device == "cpu"
        assert helper.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert helper.temperature == 0.5
        assert os.environ["HF_HOME"] == "/tmp/test_cache"
        mock_tokenizer.from_pretrained.assert_called_once_with("HuggingFaceTB/SmolLM2-135M-Instruct")
        mock_model.from_pretrained.assert_called_once()
        mock_model_instance.eval.assert_called_once()

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_initialization_cuda(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Test LLMHelper initialization with CUDA."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = True

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Test initialization with CUDA
        config = {"model": "HuggingFaceTB/SmolLM2-135M-Instruct"}
        helper = LLMHelper(config=config)

        assert helper.device == "cuda"

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_default_config(self, mock_torch, mock_model, mock_tokenizer):
        """Test LLMHelper with default configuration."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Test initialization with default config (None)
        helper = LLMHelper()

        assert helper.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert helper.temperature == 1.0
        assert os.environ["HF_HOME"] == "./.cache/"

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_empty_config(self, mock_torch, mock_model, mock_tokenizer):
        """Test LLMHelper with empty configuration."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Test initialization with empty config
        helper = LLMHelper(config={})

        assert helper.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert helper.temperature == 1.0

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_generate(self, mock_torch, mock_model, mock_tokenizer):
        """Test text generation."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock chat template application
        mock_chat_input = MagicMock()
        mock_chat_input.to.return_value = mock_chat_input
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input

        # Mock decode
        mock_tokenizer_instance.decode.return_value = "Generated SQL response"

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock model generation
        mock_outputs = MagicMock()
        mock_outputs.to.return_value = mock_outputs
        mock_outputs.__getitem__.return_value = "output_tensor"
        mock_model_instance.generate.return_value = mock_outputs

        # Initialize helper
        config = {"model": "HuggingFaceTB/SmolLM2-135M-Instruct", "parameters": {"temperature": 0.8}}
        helper = LLMHelper(config=config)

        # Test generation
        result = helper.generate("Generate SQL for user data")

        assert result == "Generated SQL response"
        mock_tokenizer_instance.apply_chat_template.assert_called_once()
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer_instance.decode.assert_called_once()

        # Verify the system message is included
        chat_template_call = mock_tokenizer_instance.apply_chat_template.call_args[0][0]
        assert len(chat_template_call) == 2  # system + user messages
        assert chat_template_call[0]["role"] == "system"
        assert "SQL" in chat_template_call[0]["content"]
        assert chat_template_call[1]["role"] == "user"
        assert chat_template_call[1]["content"] == "Generate SQL for user data"

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    def test_llm_helper_initialization_error(self, mock_model, mock_tokenizer):
        """Test LLMHelper initialization error handling."""
        # Mock tokenizer to raise exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")

        # Test that ValueError is raised
        config = {"model": "invalid-model"}
        with pytest.raises(ValueError, match="Model invalid-model is not compatible. Supported models: .*$"):
            LLMHelper(config=config)

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_generate_with_kwargs(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Test text generation with custom kwargs."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.decode.return_value = "Generated response with kwargs"

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock tensor operations
        mock_chat_input = MagicMock()
        mock_chat_input.to.return_value = mock_chat_input
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input

        mock_outputs = MagicMock()
        mock_outputs.to.return_value = mock_outputs
        mock_outputs.__getitem__.return_value = "output_tensor"
        mock_model_instance.generate.return_value = mock_outputs

        # Initialize helper
        config = {"model": "HuggingFaceTB/SmolLM2-135M-Instruct"}
        helper = LLMHelper(config=config)

        # Test generation with custom parameters
        result = helper.generate("Test prompt", max_length=150, temperature=0.5)

        assert result == "Generated response with kwargs"

        # Check that generate was called with parameters
        generate_call_args = mock_model_instance.generate.call_args
        assert generate_call_args is not None
        # The method uses max_new_tokens=150 and temperature from config
        assert "max_new_tokens" in generate_call_args[1]
        assert generate_call_args[1]["max_new_tokens"] == 150

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_generate_default_parameters(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Test text generation with default parameters."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.decode.return_value = "Default response"

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock tensor operations
        mock_chat_input = MagicMock()
        mock_chat_input.to.return_value = mock_chat_input
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input

        mock_outputs = MagicMock()
        mock_outputs.to.return_value = mock_outputs
        mock_outputs.__getitem__.return_value = "output_tensor"
        mock_model_instance.generate.return_value = mock_outputs

        # Initialize helper with custom temperature
        config = {"parameters": {"temperature": 0.9}}
        helper = LLMHelper(config=config)

        # Test generation without custom parameters
        result = helper.generate("Test prompt")

        assert result == "Default response"
        assert helper.temperature == 0.9

        # Check that generate was called with default max_new_tokens
        generate_call_args = mock_model_instance.generate.call_args
        assert "max_new_tokens" in generate_call_args[1]
        assert generate_call_args[1]["max_new_tokens"] == 150
        assert generate_call_args[1]["temperature"] == 0.9

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_system_prompt(self, mock_torch, mock_model, mock_tokenizer):
        """Test that the system prompt is correctly set for SQL generation."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock tensor operations
        mock_chat_input = MagicMock()
        mock_chat_input.to.return_value = mock_chat_input
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input

        mock_outputs = MagicMock()
        mock_outputs.to.return_value = mock_outputs
        mock_outputs.__getitem__.return_value = "output_tensor"
        mock_model_instance.generate.return_value = mock_outputs
        mock_tokenizer_instance.decode.return_value = "SQL response"

        # Initialize helper
        helper = LLMHelper()

        # Test generation
        helper.generate("Create a query")

        # Verify the messages structure
        call_args = mock_tokenizer_instance.apply_chat_template.call_args
        messages = call_args[0][0]
        
        # Check system message
        assert messages[0]["role"] == "system"
        assert "SQL" in messages[0]["content"]
        assert "deviate" in messages[0]["content"]
        
        # Check user message
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Create a query"

    @patch("app.llm_helper.AutoTokenizer")
    @patch("app.llm_helper.AutoModelForCausalLM")
    @patch("app.llm_helper.torch")
    def test_llm_helper_cache_directory(self, mock_torch, mock_model, mock_tokenizer):
        """Test that cache directory is properly set."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Test with custom cache directory
        custom_cache = "/tmp/custom_cache"
        config = {"cache_dir": custom_cache,"model": "HuggingFaceTB/SmolLM2-135M-Instruct"}

        helper = LLMHelper(config=config)  # noqa: F841
        
        assert os.environ["HF_HOME"] == custom_cache
