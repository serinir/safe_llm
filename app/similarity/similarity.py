"""
Similarity calculation module for text comparison.
"""

from typing import Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# class SimilarityCalculator(ABC):
#     """Abstract base class for similarity calculators."""

#     @abstractmethod
#     def calculate(self, text1: str, text2: str) -> float:
#         """Calculate similarity between two texts."""
#         pass

#     @property
#     @abstractmethod
#     def method_name(self) -> str:
#         """Return the name of the similarity method."""
#         pass


# class JaccardSimilarity(SimilarityCalculator):
#     """Jaccard similarity based on word overlap."""

#     def calculate(self, text1: str, text2: str) -> float:
#         """Calculate Jaccard similarity between two texts."""
#         words1 = set(text1.lower().split())
#         words2 = set(text2.lower().split())

#         if not words1 and not words2:
#             return 1.0
#         elif not words1 or not words2:
#             return 0.0

#         intersection = words1.intersection(words2)
#         union = words1.union(words2)
#         return len(intersection) / len(union)

#     @property
#     def method_name(self) -> str:
#         return "jaccard_word_overlap"


# class CosineTfIdfSimilarity(SimilarityCalculator):
#     """Cosine similarity using TF-IDF vectors."""

#     def calculate(self, text1: str, text2: str) -> float:
#         """Calculate cosine similarity using TF-IDF vectors."""
#         try:
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             from sklearn.metrics.pairwise import cosine_similarity

#             vectorizer = TfidfVectorizer()
#             tfidf_matrix = vectorizer.fit_transform([text1, text2])

#             similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
#             return float(similarity_score)

#         except ImportError:
#             logger.warning(
#                 "scikit-learn not available, falling back to Jaccard similarity"
#             )
#             fallback = JaccardSimilarity()
#             return fallback.calculate(text1, text2)

#     @property
#     def method_name(self) -> str:
#         return "cosine_tfidf"


class SimilarityService:
    def __init__(self):
        self.calculators = {
            "jaccard": self.jaccard_similarity,
            "cosine_tfidf": self.tf_idf_similarity,
        }
        self.default_method = "cosine_tfidf"

    def calculate_similarity(
        self, text1: str, text2: str, method: str = None
    ) -> Tuple[float, str]:
        """Calculate similarity between two texts using specified method."""
        if method is None:
            method = self.default_method

        if method not in self.calculators:
            raise ValueError(f"Unknown similarity method: {method}")

        similarity_score = self.calculators[method](text1, text2)
        return similarity_score, method

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def tf_idf_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return float(similarity_score)

    def list_methods(self) -> list:
        """List all available similarity methods."""
        return list(self.calculators.keys())
