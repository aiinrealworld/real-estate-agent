FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY requirements.txt ./

# Install pip-tools and compile requirements
RUN pip install pip-tools

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV VAPI_EXPOSE_PORT=8000

# Run the application
CMD ["python", "src/voice_vapi.py"] 