FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and tools
RUN pip install --upgrade pip setuptools wheel


# Install Python dependencies
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/temp_uploads /app/model_storage

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/food_recognizer_model.json
ENV TEMP_UPLOAD_DIR=/app/temp_uploads
ENV MODEL_STORAGE_DIR=/app/model_storage

# Expose port
EXPOSE 8000

## Health check
#HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
