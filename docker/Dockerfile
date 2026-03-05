FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy only dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment inside container
RUN uv venv /app/.venv

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY app ./app

# Copy trained models
COPY ../saved_models ./saved_models

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]