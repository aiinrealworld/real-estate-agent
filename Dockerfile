# Use 3.12 to satisfy packages that require Python >=3.12
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (adjust as you need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/

EXPOSE 8000
ENV PYTHONPATH=/app
ENV VAPI_EXPOSE_PORT=8000

CMD ["python", "src/voice_vapi.py"]