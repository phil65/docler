# Docler API Server

A FastAPI server for the Docler document conversion library.

## Installation

```bash
pip install "docler[server]"
```

## Usage

```bash
docler api --port 8000                                    # localhost, no token needed
DOCLER_API_TOKEN=... docler api --host 0.0.0.0 --port 8000  # any other host needs a token
```

### Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `DOCLER_API_TOKEN` | unset | Bearer token clients send as `Authorization: Bearer <token>` (compared in constant time). Required whenever the server isn't bound to a loopback address: `docler api` refuses to start without it, and a server started otherwise (e.g. `uvicorn` directly) answers 403 to requests on a non-loopback address. |
| `DOCLER_API_CORS_ORIGINS` | unset (CORS off) | Comma-separated origins allowed to call the API from a browser. No credentials mode: the token travels in the `Authorization` header. |
| `DOCLER_API_MAX_UPLOAD_MB` | `100` | Largest request body; bigger ones get 413, checked while the body arrives (before it's parsed or stored). |

`/health` and `/ready` need no token.

### Endpoints

- **POST /v1/convert**: convert a document, and chunk it when a chunker config is given.
  Form fields: `file`, `converter` (converter config JSON), `chunker` (chunker config
  JSON, optional), `pdf_password` (optional), `images` (`inline`, the default, or `none`).
- **POST /v1/pdf/metadata**: PDF metadata (page count, encryption, document info).
  Form fields: `file`, `pdf_password` (optional).
- **GET /v1/converters**: available converters, with their config schemas.
- **GET /v1/chunkers**: available chunkers, with their config schemas.
- **GET /health**, **GET /ready**: health and readiness checks.

Configs sent by clients cannot set `base_url` or `endpoint` (the URL that receives the
server's API keys and the uploaded file): the server sets them through its environment
(`UPSTAGE_BASE_URL`, `AZURE_DOC_INTELLIGENCE_ENDPOINT`, `DOCLING_REMOTE_ENDPOINT`). Such a
config is rejected with 400.

### Response of `/v1/convert`

```json
{
  "document": {"content": "# Title\n\n![fig](img-0.png)...", "title": "doc.pdf",
               "mime_type": "application/pdf", "metadata": {}, "...": "..."},
  "images": [{"id": "img-0", "filename": "img-0.png", "mime_type": "image/png",
              "description": null, "metadata": {}, "content": "<base64>"}],
  "chunks": [{"content": "...", "chunk_index": 0, "page_number": 1,
              "image_ids": ["img-0"], "metadata": {}}]
}
```

Each image is listed once, in `images`; the markdown refers to it by `filename`, chunks by
`id` (`image_ids`). With `images=none`, `content` is `null` (ids, filenames and metadata
stay, so the markdown's references still resolve to something). `chunks` is `null` when no
chunker was given.

## Examples

### Convert a document

```bash
curl -X POST "http://localhost:8000/v1/convert" \
  -H "Authorization: Bearer $DOCLER_API_TOKEN" \
  -F "file=@document.pdf" \
  -F 'converter={"type": "mistral"}'
```

### Convert an encrypted PDF, without image data

```bash
curl -X POST "http://localhost:8000/v1/convert" \
  -H "Authorization: Bearer $DOCLER_API_TOKEN" \
  -F "file=@encrypted-document.pdf" \
  -F 'converter={"type": "mistral"}' \
  -F "pdf_password=your-pdf-password" \
  -F "images=none"
```

### Convert and chunk a document

```bash
curl -X POST "http://localhost:8000/v1/convert" \
  -H "Authorization: Bearer $DOCLER_API_TOKEN" \
  -F "file=@document.pdf" \
  -F 'converter={"type": "mistral"}' \
  -F 'chunker={"type": "markdown", "max_chunk_size": 1000}'
```

### Get PDF metadata

```bash
curl -X POST "http://localhost:8000/v1/pdf/metadata" \
  -H "Authorization: Bearer $DOCLER_API_TOKEN" \
  -F "file=@document.pdf"
```

## Development

```bash
docler api --reload
```
