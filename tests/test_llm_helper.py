"""
Tests for LLM helper functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from app.llm_helper import LLMHelper


class TestLLMHelper:
    """Test the LLMHelper class."""

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    @patch('app.llm_helper.torch')
    def test_llm_helper_initialization_cpu(self, mock_torch, mock_model, mock_tokenizer):
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
        
        # Test initialization
        helper = LLMHelper("test-model")
        
        assert helper.device == "cpu"
        assert helper.model_name == "test-model"
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once()
        mock_model_instance.eval.assert_called_once()

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    @patch('app.llm_helper.torch')
    def test_llm_helper_initialization_cuda(self, mock_torch, mock_model, mock_tokenizer):
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
        
        # Test initialization
        helper = LLMHelper("test-model")
        
        assert helper.device == "cuda"

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    @patch('app.llm_helper.torch')
    def test_llm_helper_default_model(self, mock_torch, mock_model, mock_tokenizer):
        """Test LLMHelper with default model."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Test initialization with default model
        helper = LLMHelper()
        
        assert helper.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    @patch('app.llm_helper.torch')
    def test_llm_helper_generate(self, mock_torch, mock_model, mock_tokenizer):
        """Test text generation."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock chat template
        mock_chat_input = {"input_ids": "mocked_tensor"}
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input
        
        # Mock decode
        mock_tokenizer_instance.decode.return_value = "Generated response"
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.generate.return_value = ["generated_output"]
        
        # Mock tensor operations
        mock_chat_input_tensor = MagicMock()
        mock_chat_input_tensor.to.return_value = mock_chat_input_tensor
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input_tensor
        
        mock_outputs = MagicMock()
        mock_outputs.to.return_value = ["output_tensor"]
        mock_model_instance.generate.return_value = mock_outputs
        
        # Initialize helper
        helper = LLMHelper("test-model")
        
        # Test generation
        result = helper.generate("Test prompt")
        
        assert result == "Generated response"
        mock_tokenizer_instance.apply_chat_template.assert_called_once()
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer_instance.decode.assert_called_once()

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    def test_llm_helper_initialization_error(self, mock_model, mock_tokenizer):
        """Test LLMHelper initialization error handling."""
        # Mock tokenizer to raise exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        # Test that RuntimeError is raised
        with pytest.raises(RuntimeError, match="Failed to load model"):
            LLMHelper("invalid-model")

    @patch('app.llm_helper.AutoTokenizer')
    @patch('app.llm_helper.AutoModelForCausalLM')
    @patch('app.llm_helper.torch')
    def test_llm_helper_generate_with_kwargs(self, mock_torch, mock_model, mock_tokenizer):
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
        mock_chat_input_tensor = MagicMock()
        mock_chat_input_tensor.to.return_value = mock_chat_input_tensor
        mock_tokenizer_instance.apply_chat_template.return_value = mock_chat_input_tensor
        
        mock_outputs = MagicMock()
        mock_outputs.to.return_value = ["output_tensor"]
        mock_model_instance.generate.return_value = mock_outputs
        
        # Initialize helper
        helper = LLMHelper("test-model")
        
        # Test generation with custom parameters
        result = helper.generate("Test prompt", temperature=0.5, max_length=100)
        
        assert result == "Generated response with kwargs"
        
        # Check that generate was called with custom parameters
        generate_call_args = mock_model_instance.generate.call_args
        assert "max_new_tokens" in generate_call_args[1]
        assert generate_call_args[1]["max_new_tokens"] == 150
