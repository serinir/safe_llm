FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ADD . /app
WORKDIR /app
RUN uv sync --locked
CMD ["uv", "run", "__main__.py"]
