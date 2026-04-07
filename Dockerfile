# ---------------------------------------------------------------------------
# Email Management RL Environment
# ---------------------------------------------------------------------------
# Supports two run modes (set via RUN_MODE env var):
#   inference  – run the baseline inference script (default)
#   app        – launch the Gradio web UI on port 7860
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Non-interactive apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Expose port (for Gradio UI / HuggingFace Spaces)
EXPOSE 7860

# Runtime environment variable defaults (override with -e at docker run)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-3.5-turbo"
# OpenEnv API Server requires FastAPI & Uvicorn (see requirements.txt)
# The entrypoint launches the Environment Server which listens for /reset and /step
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
