from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router as api_router
from app.utils import load_config
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safe LLM Endpoint",
    version="0.0.1",
    description="A secure FastAPI endpoint for LLM interactions with guardrails and similarity checking",
)

try:
    config = load_config("config.json")
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = {}

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["API"])


@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {
        "message": "Safe LLM Endpoint API",
        "version": "0.0.1",
        "docs": "/docs",
        "api_prefix": "/api",
    }
