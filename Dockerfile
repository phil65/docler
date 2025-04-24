# Build stage for installing dependencies
FROM python:3.13-slim-bookworm AS builder

# Add build argument with default value
ARG EXTRAS="server,light"

WORKDIR /build

# Install system dependencies and curl for UV
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    poppler-utils \
    # Only install heavy OCR dependencies if needed
    $(if echo "${EXTRAS}" | grep -q "all\|marker\|docling\|markitdown"; then \
        echo "tesseract-ocr libtesseract-dev libpoppler-cpp-dev"; \
    else \
        echo ""; \
    fi) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV for faster dependency resolution
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy the entire project first (needed for version detection)
COPY . .

# Install dependencies with UV using the build arg
RUN uv pip install --system ".[${EXTRAS}]"

# Final stage with minimal runtime image
FROM python:3.13-slim-bookworm

# Pass build arg to final stage
ARG EXTRAS="server,light"

WORKDIR /app

# Install only runtime dependencies, conditionally based on extras
RUN apt-get update && apt-get install --no-install-recommends -y \
    poppler-utils \
    # Only install heavy OCR dependencies if needed
    $(if echo "${EXTRAS}" | grep -q "all\|marker\|docling\|markitdown"; then \
        echo "tesseract-ocr libtesseract-dev"; \
    else \
        echo ""; \
    fi) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages and binaries
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Store which extras were installed for potential runtime checks
ENV INSTALLED_EXTRAS="${EXTRAS}"
ENV PORT="8000"

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "python -m docler_api api --host 0.0.0.0 --port ${PORT}"]
