# Safe LLM Endpoint

A secure FastAPI endpoint for LLM interactions with guardrails and similarity checking.

## Features

- **Guardrails**: Input and output validation using configurable rules (pattern matching, length validation)
- **LLM Integration**: Built-in support for HuggingFace transformers with device auto-detection (CPU/CUDA/MPS)
- **Similarity Checking**: Multiple methods for text similarity calculation (Jaccard, Cosine TF-IDF)
- **Intelligent Caching**: Automatic caching of LLM predictions with similarity-based cache lookup
- **Production Ready**: Docker support, comprehensive testing, and proper error handling

## Project Structure

```
safe_llm_endpoint/
├── __main__.py              # Application entry point
├── main.py                  # FastAPI application setup
├── config.json              # Configuration file
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Dependency lock file
├── Dockerfile               # Docker container configuration
├── run-tests.sh             # Test runner script
└── app/
    ├── routes.py            # API route definitions and main logic
    ├── models.py            # Pydantic models for requests/responses
    ├── utils.py             # Utility functions (config loading)
    ├── llm_helper.py        # LLM integration with HuggingFace transformers
    ├── guardrails/
    │   ├── __init__.py
    │   └── guardrail.py     # GuardrailService for input/output validation
    └── similarity/
        ├── __init__.py
        └── similarity.py    # SimilarityService with multiple algorithms
└── tests/
    ├── conftest.py          # Test configuration and fixtures
    ├── test_api.py          # API endpoint integration tests
    ├── test_guardrails.py   # Guardrail service unit tests
    ├── test_similarity.py   # Similarity service unit tests
    ├── test_llm_helper.py   # LLM helper unit tests
    └── test_utils.py        # Utility function tests
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

- `POST /api/input-guardrail` - Validate input text against configured rules
- `POST /api/output-guardrail` - Validate generated output text
### Similarity Endpoints

- `POST /api/similarity` - Calculate text similarity using specified method
- `GET /api/similarity/methods` - List available similarity calculation methods

### LLM Endpoints

- `POST /api/prediction` - Generate text predictions using configured LLM with intelligent caching

## Configuration

The `config.json` file defines guardrails, similarity settings, and LLM configuration:

```json
{
    "guardrails": [
        {
            "name": "BasicInputGuardrail",
            "guardrail_type": "input",
            "description": "Basic input validation",
            "rules": [
                {
                    "type": "length",
                    "min_length": 5,
                    "max_length": 100
                },
                {
                    "type": "pattern",
                    "pattern": "[^a-zA-Z0-9\\s]",
                    "replace_with": ""
                }
            ]
        },
        {
            "name": "OutputGuardrail",
            "guardrail_type": "output",
            "description": "Output validation",
            "rules": [
                {
                    "type": "length",
                    "min_length": 1,
                    "max_length": 500
                }
            ]
        }
    ],
    "prediction": {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "cache_dir": "./.cache/",
        "parameters": {
            "temperature": 0.8
        }
    }
}
```

### Rule Types

- **length**: Validate text length with `min_length` and/or `max_length`
- **pattern**: Clean text using regex pattern with optional `replace_with`
- **llm**: LLM-based validation (future enhancement)

### LLM Configuration

- **model**: HuggingFace model identifier
- **cache_dir**: Directory for model caching
- **parameters**: Generation parameters (temperature, etc.)

## Similarity Methods

- **jaccard**: Jaccard similarity based on word overlap
- **cosine_tfidf**: Cosine similarity using TF-IDF vectors

## Development

The application uses a modular, service-oriented architecture:

- **GuardrailService**: Handles input/output validation with configurable rules
- **SimilarityService**: Manages different similarity calculation methods
- **LLMHelper**: Integrates with HuggingFace transformers for text generation
- **Routes**: Clean API layer that delegates to appropriate services
- **Models**: Pydantic models for request/response validation

### Key Components

- **Intelligent Caching**: Uses similarity checking to find cached predictions for similar inputs
- **Device Auto-detection**: Automatically uses CUDA, MPS (Apple Silicon), or CPU
- **Modular Rules**: Easy to extend guardrail rules and similarity methods
- **Comprehensive Testing**: Unit tests, integration tests, and mocking for all components

To add new features:
- **Guardrail rules**: Extend `GuardrailService` in `app/guardrails/guardrail.py`
- **Similarity methods**: Add methods to `SimilarityService` in `app/similarity/similarity.py`  
- **LLM models**: Configure different models in `config.json`

## Testing

Comprehensive test suite located in the `tests/` directory:

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest --cov=app tests/

# Run specific test categories
./run-tests.sh unit          # Unit tests only
./run-tests.sh integration   # Integration tests only
./run-tests.sh coverage      # Tests with coverage
```

### Test Categories

- **Unit Tests**: Individual component testing with comprehensive mocking
- **Integration Tests**: Full API workflow testing with real HTTP requests
- **API Tests**: Complete endpoint testing including error handling
- **Service Tests**: GuardrailService, SimilarityService, and LLMHelper testing

### Test Features

- **Comprehensive Mocking**: LLM models mocked to avoid loading during tests
- **Fixture Management**: Clean test data and configuration setup
- **Edge Case Coverage**: Empty inputs, invalid data, error conditions
- **Performance Testing**: Concurrent request handling and large input testing
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
