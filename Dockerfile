# Multi-stage Docker build for Iris Classification Model

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app.py .
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/
COPY notebooks/ notebooks/
COPY README.md .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Streamlit configuration
RUN mkdir -p ~/.streamlit && \
    echo "[logger]" > ~/.streamlit/config.toml && \
    echo "level = \"error\"" >> ~/.streamlit/config.toml && \
    echo "[client]" >> ~/.streamlit/config.toml && \
    echo "showErrorDetails = false" >> ~/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
