"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from unittest.mock import patch, MagicMock


class TestRootEndpoint:
    """Test the root API endpoint."""

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/api/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "available_endpoints" in data
        assert data["version"] == "0.0.1"

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "message" in data


class TestGuardrailEndpoints:
    """Test guardrail API endpoints."""

    def test_input_guardrail_valid(self, client, sample_texts, config):
        """Test input guardrail with valid text."""
        response = client.post(
            "/api/input-guardrail", json={"text": sample_texts["valid_short"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_valid" in data
        assert "message" in data
        assert "guardrail_used" in data

    def test_input_guardrail_invalid(self, client, sample_texts):
        """Test input guardrail with invalid text."""
        response = client.post(
            "/api/input-guardrail", json={"text": sample_texts["too_long"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_valid" in data
        assert "message" in data
        assert "guardrail_used" in data

    def test_output_guardrail(self, client, sample_texts):
        """Test output guardrail endpoint."""
        response = client.post(
            "/api/output-guardrail", json={"text": sample_texts["valid_long"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_valid" in data
        assert "message" in data
        assert "guardrail_used" in data


class TestSimilarityEndpoints:
    """Test similarity API endpoints."""

    def test_similarity_endpoint(self, client, sample_texts):
        """Test similarity calculation endpoint."""
        response = client.post(
            "/api/similarity",
            json={
                "text1": sample_texts["similar_text1"],
                "text2": sample_texts["similar_text2"],
                "method": "jaccard",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "similarity_score" in data
        assert "method_used" in data
        assert 0 <= data["similarity_score"] <= 1

    def test_similarity_methods_endpoint(self, client):
        """Test listing similarity methods."""
        response = client.get("/api/similarity/methods")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0


class TestPredictionEndpoint:
    """Test prediction API endpoint."""

    def test_prediction_endpoint(self, client, sample_texts):
        """Test prediction endpoint."""
        response = client.post(
            "/api/prediction",
            json={
                "input_text": sample_texts["valid_short"],
                "model_name": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data


class TestErrorHandling:
    """Test error handling in API endpoints."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post("/api/input-guardrail", data="invalid json")

        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/api/input-guardrail",
            json={},  # Missing required 'text' field
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_nonexistent_endpoint(self, client):
        """Test handling of non-existent endpoints."""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_empty_text_input_guardrail(self, client):
        """Test input guardrail with empty text."""
        response = client.post("/api/input-guardrail", json={"text": ""})
        
        assert response.status_code == 200
        data = response.json()
        # Empty text should fail minimum length validation
        assert not data["is_valid"]

    def test_very_short_text_input_guardrail(self, client):
        """Test input guardrail with very short text."""
        response = client.post("/api/input-guardrail", json={"text": "hi"})
        
        assert response.status_code == 200
        data = response.json()
        # Very short text should fail minimum length validation (< 5 chars)
        assert not data["is_valid"]

    def test_prediction_with_invalid_request(self, client):
        """Test prediction endpoint with invalid request."""
        response = client.post("/api/prediction", json={})
        
        assert response.status_code == 422  # Missing required fields

    def test_similarity_with_missing_fields(self, client):
        """Test similarity endpoint with missing fields."""
        response = client.post("/api/similarity", json={"text1": "only one text"})
        
        assert response.status_code == 422  # Missing text2 and method

    def test_similarity_with_invalid_method(self, client, sample_texts):
        """Test similarity endpoint with invalid method."""
        response = client.post(
            "/api/similarity",
            json={
                "text1": sample_texts["similar_text1"],
                "text2": sample_texts["similar_text2"],
                "method": "invalid_method",
            },
        )
        
        # Should return 500 
        assert response.status_code == 500


class TestAdditionalEndpoints:
    """Test additional API functionality for better coverage."""

    def test_prediction_with_special_characters_cleaned(self, client):
        """Test prediction with text that gets cleaned by guardrails."""
        response = client.post(
            "/api/prediction",
            json={
                "input_text": "Hello @#$% world!",  # Contains special chars
                "model_name": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data

    def test_output_guardrail_with_too_long_text(self, client):
        """Test output guardrail with text exceeding max length."""
        long_text = "A" * 150  # Exceeds max_length of 100
        response = client.post("/api/output-guardrail", json={"text": long_text})

        assert response.status_code == 200
        data = response.json()
        # Should pass because long text gets truncated
        assert data["is_valid"]

    def test_output_guardrail_with_empty_text(self, client):
        """Test output guardrail with empty text."""
        response = client.post("/api/output-guardrail", json={"text": ""})

        assert response.status_code == 200
        data = response.json()
        print(data)
        # With properly configured output guardrail, empty text should fail minimum length validation
        assert not data["is_valid"]
        assert data["guardrail_used"] == "TestOutputGuardrail"

    def test_similarity_cosine_method(self, client, sample_texts):
        """Test similarity with cosine method."""
        response = client.post(
            "/api/similarity",
            json={
                "text1": sample_texts["similar_text1"],
                "text2": sample_texts["similar_text2"],
                "method": "cosine_tfidf",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "similarity_score" in data
        assert "method_used" in data
        assert 0 <= data["similarity_score"] <= 1

    def test_similarity_with_identical_texts(self, client):
        """Test similarity with identical texts."""
        text = "This is identical text"
        response = client.post(
            "/api/similarity",
            json={
                "text1": text,
                "text2": text,
                "method": "jaccard",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Identical texts should have high similarity
        assert data["similarity_score"] >= 0.9

    def test_similarity_with_completely_different_texts(self, client):
        """Test similarity with completely different texts."""
        response = client.post(
            "/api/similarity",
            json={
                "text1": "apple banana orange",
                "text2": "car truck motorcycle",
                "method": "jaccard",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Completely different texts should have low similarity
        assert data["similarity_score"] <= 0.5

    def test_prediction_endpoint_different_parameters(self, client, sample_texts):
        """Test prediction endpoint with different parameters."""
        response = client.post(
            "/api/prediction",
            json={
                "input_text": sample_texts["valid_long"],
                "model_name": "different_model",
                "temperature": 0.5,
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data

    def test_cache_behavior_prediction(self, client, sample_texts):
        """Test that cache works for repeated prediction requests."""
        request_data = {
            "input_text": sample_texts["valid_short"],
            "model_name": "test_model",
        }
        
        # First request
        response1 = client.post("/api/prediction", json=request_data)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second identical request (should use cache)
        response2 = client.post("/api/prediction", json=request_data)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Both should return prediction
        assert "prediction" in data1
        assert "prediction" in data2
