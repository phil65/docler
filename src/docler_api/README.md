# Docler API Server

A FastAPI server for the Docler document conversion library.

## Installation

```bash
pip install "docler[server]"
```

## Usage

### Start the API server

```bash
docler-api api --host 0.0.0.0 --port 8000
```

### API Endpoints

- **GET /** - Root endpoint, returns welcome message and version
- **POST /api/convert** - Convert a document to markdown (images included as base64)
- **POST /api/chunk** - Convert and chunk a document (images included as base64)
- **GET /api/converters** - List all available converters
- **GET /api/chunkers** - List all available chunkers
- **POST /api/pdf/metadata** - Get PDF metadata including page count
- **GET /health** - Health check endpoint
- **GET /ready** - Readiness check endpoint

## Examples

### Convert a document

```bash
curl -X POST "http://localhost:8000/api/convert" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "config={\"type\":\"marker\",\"dpi\":300}"
```

The response will include all extracted images as base64-encoded data in the JSON response.

### Convert an encrypted PDF

```bash
curl -X POST "http://localhost:8000/api/convert" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@encrypted-document.pdf" \
  -F "config={\"type\":\"marker\",\"dpi\":300}" \
  -F "pdf_password=your-pdf-password"
```

### Convert and chunk a document

```bash
curl -X POST "http://localhost:8000/api/chunk" \
  -H "Content-Type: application/json" \
  -F "file=@document.pdf" \
  -d '{
    "converter_config": {"type": "marker", "dpi": 300},
    "chunker_config": {"type": "markdown", "max_chunk_size": 1000}
  }'
```

Images are automatically included as base64-encoded data in both the document and individual chunks.

### Get PDF metadata (including encrypted PDFs)

```bash
# Check if PDF is encrypted
curl -X POST "http://localhost:8000/api/pdf/metadata" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Get metadata from encrypted PDF
curl -X POST "http://localhost:8000/api/pdf/metadata" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@encrypted-document.pdf" \
  -F "pdf_password=your-pdf-password"
```

## Development

To run the server with auto-reload during development:

```bash
docler-api api --reload
```
