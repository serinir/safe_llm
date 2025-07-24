# Safe LLM Endpoint

A secure FastAPI endpoint for LLM interactions with guardrails and similarity checking.

## Features

- **Guardrails**: Input and output validation using configurable rules
- **Similarity Checking**: Multiple methods for text similarity calculation
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive API**: RESTful endpoints with automatic documentation

## Project Structure

```
safe_llm_endpoint/
├── __main__.py              # Application entry point
├── config.json              # Configuration file
├── logging.conf             # Logging configuration
├── pyproject.toml           # Project dependencies
├── start.sh                 # Startup script
└── app/
    ├── routes.py            # API route definitions
    ├── utils.py             # Utility functions
    ├── guardrails/
    │   ├── __init__.py
    │   └── guardrail.py     # Guardrail classes and manager
    └── similarity/
        ├── __init__.py
        └── similarity.py    # Similarity calculation service
```

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the application:
   ```bash
   ./start.sh
   # or
   uv run python __main__.py
   ```

## API Endpoints

### Core Endpoints

- `GET /api/` - API information and available services
- `GET /api/health` - Health check

### Guardrail Endpoints

- `POST /api/input-guardrail` - Validate input text
- `POST /api/output-guardrail` - Validate output text
- `GET /api/guardrails` - List available guardrails
- `GET /api/guardrails/{name}` - Get guardrail information

### Similarity Endpoints

- `POST /api/similarity` - Calculate text similarity
- `GET /api/similarity/methods` - List available similarity methods

### Other Endpoints

- `POST /api/prediction` - Generate predictions (placeholder)

## Configuration

The `config.json` file defines guardrails with their rules:

```json
{
    "guardrails": [
        {
            "name": "BasicGuardrail",
            "guardrail_type": "input",
            "description": "Basic input validation",
            "rules": [
                {
                    "type": "length",
                    "max_length": 100,
                    "error_message": "Input too long"
                },
                {
                    "type": "pattern",
                    "pattern": "^[a-zA-Z0-9\\s]+$",
                    "error_message": "Invalid characters"
                }
            ]
        }
    ]
}
```

### Rule Types

- **length**: Validate text length (min_length, max_length)
- **pattern**: Validate against regex pattern
- **llm**: LLM-based validation (coming soon)

## Similarity Methods

- **jaccard**: Jaccard similarity based on word overlap
- **cosine_tfidf**: Cosine similarity using TF-IDF vectors

## Development

The application uses a modular architecture:

- **GuardrailService**: Handles all guardrail operations
- **SimilarityService**: Manages different similarity calculation methods
- **Routes**: Clean API layer that delegates to appropriate services

To add new guardrail rules or similarity methods, extend the respective classes in their modules.

## Testing
Unit tests are located in the `tests/` directory. Use `pytest` to run them:

```bash
uv run pytest tests/
```
to generate a coverage report, use:

```bash
uv run pytest --cov=app tests/
```
Integration tests not included yet, but can be added in the future.
## Building container
To build a Docker container for the application, use the provided `Dockerfile`:

```bash
docker build -t safe_llm_endpoint .
```
To run the container:

```bash
docker run -p 8000:8000 safe_llm_endpoint
```

## API Documentation

When running, visit:
- http://localhost:8000/docs - Interactive API documentation
- http://localhost:8000/redoc - Alternative documentation format
