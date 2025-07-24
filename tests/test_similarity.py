"""
Tests for similarity functionality.
"""

import pytest
from app.similarity import SimilarityService


class TestJaccardSimilarity:
    """Test the Jaccard similarity calculator."""

    def test_identical_texts(self):
        """Test similarity of identical texts."""
        calculator = SimilarityService()
        text = "hello world"

        similarity, _ = calculator.calculate_similarity(text, text, "jaccard")
        assert similarity == 1.0

    def test_completely_different_texts(self):
        """Test similarity of completely different texts."""
        calculator = SimilarityService()

        similarity, _ = calculator.calculate_similarity(
            "hello world", "python programming", "jaccard"
        )
        assert similarity == 0.0

    def test_partially_similar_texts(self, sample_texts):
        """Test similarity of partially similar texts."""
        calculator = SimilarityService()

        similarity, _ = calculator.calculate_similarity(
            sample_texts["similar_text1"], sample_texts["similar_text2"], "jaccard"
        )

        assert 0 < similarity < 1

    def test_empty_texts(self):
        """Test similarity with empty texts."""
        calculator = SimilarityService()

        # Both empty
        similarity, _ = calculator.calculate_similarity("", "", "jaccard")
        assert similarity == 1.0

        # One empty
        similarity, _ = calculator.calculate_similarity("", "hello", "jaccard")
        assert similarity == 0.0

        similarity, _ = calculator.calculate_similarity("hello", "", "jaccard")
        assert similarity == 0.0


class TestCosineTfIdfSimilarity:
    """Test the Cosine TF-IDF similarity calculator."""

    def test_identical_texts(self):
        """Test similarity of identical texts."""
        calculator = SimilarityService()
        text = "hello world programming"

        similarity, _ = calculator.calculate_similarity(text, text, "cosine_tfidf")
        # Should be very close to 1.0 (allowing for floating point precision)
        assert abs(similarity - 1.0) < 0.001

    def test_similar_texts(self, sample_texts):
        """Test similarity of similar texts."""
        calculator = SimilarityService()

        similarity, _ = calculator.calculate_similarity(
            sample_texts["similar_text1"], sample_texts["similar_text2"], "cosine_tfidf"
        )

        # Should be relatively high similarity
        assert similarity > 0.5

    def test_different_texts(self, sample_texts):
        """Test similarity of different texts."""
        calculator = SimilarityService()

        similarity, _ = calculator.calculate_similarity(
            sample_texts["similar_text1"],
            sample_texts["different_text"],
            "cosine_tfidf",
        )

        # Should be lower similarity
        assert similarity < 0.5


class TestSimilarityService:
    """Test the SimilarityService class."""

    def test_service_creation(self):
        """Test creating a similarity service."""
        service = SimilarityService()

        assert "jaccard" in service.calculators
        assert "cosine_tfidf" in service.calculators
        assert service.default_method == "cosine_tfidf"

    def test_calculate_similarity_default(self, sample_texts):
        """Test similarity calculation with default method."""
        service = SimilarityService()

        similarity, method = service.calculate_similarity(
            sample_texts["similar_text1"], sample_texts["similar_text2"]
        )

        assert 0 <= similarity <= 1
        assert method == "cosine_tfidf"

    def test_calculate_similarity_unknown_method(self, sample_texts):
        """Test similarity calculation with unknown method."""
        service = SimilarityService()

        with pytest.raises(ValueError):
            similarity, method = service.calculate_similarity(
                sample_texts["similar_text1"],
                sample_texts["similar_text2"],
                method="unknown_method",
            )

    def test_list_methods(self):
        """Test listing available methods."""
        service = SimilarityService()

        methods = service.list_methods()
        assert "jaccard" in methods
        assert "cosine_tfidf" in methods
        assert len(methods) == 2
