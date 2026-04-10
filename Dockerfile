# # ---------------------------------------------------------------------------
# # Email Management RL Environment
# # ---------------------------------------------------------------------------
# # The container runs the FastAPI OpenEnv server (server.py) which exposes:
# #   POST /reset  – initialise a new episode
# #   POST /step   – advance the environment by one step
# # All scores are strictly in (0, 1) — never 0.0 or 1.0.
# # ---------------------------------------------------------------------------

# FROM python:3.11-slim

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# # System deps
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Copy project source
# COPY . .

# # Expose port
# EXPOSE 7860

# # Runtime defaults
# ENV API_BASE_URL="https://api.openai.com/v1"
# ENV MODEL_NAME="gpt-3.5-turbo"

# # Health check so HF Spaces knows when the server is ready
# HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
#     CMD curl -f http://localhost:7860/ || exit 1

# # Always run the FastAPI server — this is what the OpenEnv validator calls
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]




# ---------------------------------------------------------------------------
# Email Management RL Environment
# ---------------------------------------------------------------------------
# The container runs the FastAPI OpenEnv server (server.py) which exposes:
#   POST /reset  – initialise a new episode
#   POST /step   – advance the environment by one step
#   GET  /state  – return current environment state
# All scores are strictly in (0, 1) — never 0.0 or 1.0.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — install before copying source for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# ---------------------------------------------------------------------------
# HuggingFace Spaces requires a non-root user
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Runtime defaults — override these via HF Space secrets
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"
ENV HF_TOKEN=""

# Health check so HF Spaces knows when the server is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# Always run the FastAPI server — this is what the OpenEnv validator calls
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]