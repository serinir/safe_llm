"""
routers for the Safe LLM Endpoint API
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.utils import load_config
from app.guardrails.guardrail import GuardrailService
from app.similarity import SimilarityService
from app.llm_helper import LLMHelper
from app.models import (
    GuardrailRequest,
    GuardrailResponse,
    SimilarityRequest,
    SimilarityResponse,
    PredictionRequest,
    PredictionResponse,
)
import logging

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Load configuration and initialize services
input_guardrail = None
output_guardrail = None
try:
    config = load_config("config.json")
    logger.info(f"Configuration loaded successfully: {config}")
    for guardrail in config.get("guardrails", []):
        logger.info(
            f"Guardrail loaded: {guardrail['name']} of type {guardrail['guardrail_type']}"
        )
        if guardrail["guardrail_type"] == "input":
            input_guardrail = GuardrailService(guardrail)
        if guardrail["guardrail_type"] == "output":
            output_guardrail = GuardrailService(guardrail)

    similarity_service = SimilarityService()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to load configuration or initialize services: {e}")
    config = {}
    similarity_service = None
# Loading the LLM Helper outside the endpoints to avoid re-initialization
llm_helper = LLMHelper(config=config.get("prediction", {}))

cache = {}


@router.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint providing API information."""
    available_guardrails = []
    available_similarity_methods = []
    if input_guardrail:
        available_guardrails.append(input_guardrail.config["name"])
    if output_guardrail:
        available_guardrails.append(output_guardrail.config["name"])

    if similarity_service:
        available_similarity_methods = similarity_service.list_methods()

    return {
        "message": "Welcome to the Safe LLM Endpoint API!",
        "version": "0.0.1",
        "available_endpoints": [
            "/api/input-guardrail",
            "/api/output-guardrail",
            "/api/similarity",
            "/api/similarity/methods",
            "/api/prediction",
            "/api/guardrails",
            "/api/health",
        ],
        "available_guardrails": available_guardrails,
        "available_similarity_methods": available_similarity_methods,
    }


@router.post("/input-guardrail", response_model=GuardrailResponse)
async def route_input_guardrail(request: GuardrailRequest):
    """Apply input guardrails to validate text."""
    if not input_guardrail:
        return GuardrailResponse(
            is_valid=True, message="Input is valid.", guardrail_used="N/A"
        )

    try:
        result = input_guardrail.validate(request.text)
        if result.is_valid:
            return GuardrailResponse(
                is_valid=True,
                message="Input is valid",
                guardrail_used=input_guardrail.config["name"],
            )
        return GuardrailResponse(
            is_valid=result.is_valid,
            message=result.message,
            guardrail_used=input_guardrail.config["name"],
            failed_rule=result.failed_rule,
        )
    except Exception as e:
        logger.error(f"Error in input guardrail: {e}")
        return GuardrailResponse(
            is_valid=False,
            message=str(e),
            guardrail_used=input_guardrail.config["name"],
        )


@router.post("/output-guardrail", response_model=GuardrailResponse)
async def route_output_guardrail(request: GuardrailRequest):
    """Apply output guardrails to validate generated text."""
    if not output_guardrail:
        return GuardrailResponse(
            is_valid=True, message="Output is valid.", guardrail_used="N/A"
        )

    try:
        result = output_guardrail.validate(request.text)
        if result.is_valid:
            return GuardrailResponse(
                is_valid=True,
                message="Output is valid",
                guardrail_used=output_guardrail.config["name"],
            )
        return GuardrailResponse(
            is_valid=result.is_valid,
            message=result.message,
            guardrail_used=output_guardrail.config["name"],
            failed_rule=result.failed_rule,
        )
    except Exception as e:
        logger.error(f"Error in output guardrail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    """Calculate similarity between two texts using various methods."""
    if not similarity_service:
        raise HTTPException(
            status_code=500, detail="Similarity service not initialized"
        )

    try:
        similarity_score, method_used = similarity_service.calculate_similarity(
            request.text1, request.text2, request.method
        )

        return SimilarityResponse(
            similarity_score=similarity_score, method_used=method_used
        )
    except Exception as e:
        logger.error(f"Error in similarity calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prediction", response_model=PredictionResponse)
async def prediction(request: PredictionRequest):
    """Generate predictions using specified model."""
    try:
        ### check similarity with cache using similarity_service
        for key, cached_value in cache.items():
            similarity_score, _ = similarity_service.calculate_similarity(
                request.input_text, key, method="jaccard"
            )
            if similarity_score > 0.8:
                logger.info(f"Using cached prediction for: {key}")
                return PredictionResponse(prediction=cached_value)
        if input_guardrail:
            input_validation = input_guardrail.validate(request.input_text)
            if not input_validation.is_valid:
                logger.warning(f"Input validation failed: {input_validation.message}")
                return PredictionResponse(
                    prediction="Unsafe input detected, prediction not generated."
                )
        prediction_text = llm_helper.generate(request.input_text)

        if output_guardrail:
            output_validation = output_guardrail.validate(prediction_text)
            if not output_validation.is_valid:
                logger.warning(f"Output validation failed: {output_validation.message}")
                return PredictionResponse(
                    prediction="Unsafe output detected, prediction not generated."
                )
        cache[request.input_text] = prediction_text
        return PredictionResponse(prediction=prediction_text)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similarity/methods", response_model=List[str])
async def list_similarity_methods():
    """List all available similarity calculation methods."""
    if not similarity_service:
        raise HTTPException(
            status_code=500, detail="Similarity service not initialized"
        )

    return similarity_service.list_methods()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running properly"}
