#!/bin/bash

# Safe LLM Endpoint Startup Script

# Install dependencies if they're not already installed
echo "Installing dependencies..."
uv sync

# Run the FastAPI application
echo "Starting Safe LLM Endpoint API..."
uv run .
