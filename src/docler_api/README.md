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
- **POST /api/convert** - Convert a document to markdown
- **POST /api/chunk** - Convert and chunk a document
- **GET /api/converters** - List all available converters
- **GET /api/chunkers** - List all available chunkers
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

### Convert and chunk a document

```bash
curl -X POST "http://localhost:8000/api/chunk" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "converter_config={\"type\":\"marker\",\"dpi\":300}" \
  -F "chunker_config={\"type\":\"markdown\",\"max_chunk_size\":1000}"
```

## Development

To run the server with auto-reload during development:

```bash
docler-api api --reload
```
