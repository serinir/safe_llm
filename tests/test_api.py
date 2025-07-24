"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


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

    def test_input_guardrail_valid(self, client, sample_texts):
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
