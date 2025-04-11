# Build stage for installing dependencies
FROM python:3.13-slim-bookworm AS builder

WORKDIR /build

# Install system dependencies and curl for UV
RUN apt-get update && apt-get install --no-install-recommends -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-cpp-dev \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV for faster dependency resolution
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy the entire project first (needed for version detection)
COPY . .

# Install dependencies with UV (much faster than pip)
RUN uv pip install --system ".[server,all]"

# Final stage with minimal runtime image
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages and binaries
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Set environment variables for API keys and configuration
ENV OPENAI_API_KEY=""
ENV AZURE_DOC_INTELLIGENCE_ENDPOINT=""
ENV AZURE_DOC_INTELLIGENCE_KEY=""
ENV MISTRAL_API_KEY=""
ENV LLAMAPARSE_API_KEY=""
ENV PINECONE_API_KEY=""
ENV DATALAB_API_KEY=""
ENV UPSTAGE_API_KEY=""
ENV PORT="8000"

# Expose the port for FastAPI
EXPOSE 8000

# HEALTHCHECK --interval=30s --start-period=60s CMD curl -f http://localhost:${PORT}/health || exit 1
# Command to run the application
CMD ["sh", "-c", "python -m docler_api api --host 0.0.0.0 --port ${PORT}"]
