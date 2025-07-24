from pydantic import BaseModel


class GuardrailRequest(BaseModel):
    text: str


class GuardrailResponse(BaseModel):
    is_valid: bool
    message: str
    guardrail_used: str
    failed_rule: str = None


class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    method: str = "cosine_tfidf"


class SimilarityResponse(BaseModel):
    similarity_score: float
    method_used: str


class PredictionRequest(BaseModel):
    input_text: str
    model_name: str = "default"


class PredictionResponse(BaseModel):
    prediction: str
