"""FastAPI server for docler document conversion library."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mkdown import Document  # noqa: TC002

from docler import __version__ as docler_version
from docler.configs.chunker_configs import ChunkerConfig  # noqa: TC001
from docler.configs.converter_configs import ConverterConfig  # noqa: TC001
from docler_api import routes


app = FastAPI(
    title="Docler API",
    description="API for document conversion using docler",
    version=docler_version,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "Welcome to Docler API", "version": docler_version}


@app.post("/api/convert")
async def api_convert_document(
    file: UploadFile = File(..., description="The document file to convert"),  # noqa: B008
    config: str = Form(..., description="Converter configuration"),
    pdf_password: str | None = Form(None, description="Password for encrypted PDF files"),
) -> Document:
    """Convert a document file to markdown."""
    return await routes.convert_document(file, config, pdf_password)


@app.get("/api/converters")
async def api_list_converters():
    """List all available converters."""
    return await routes.list_converters()


@app.get("/api/chunkers")
async def api_list_chunkers():
    """List all available chunkers."""
    return await routes.list_chunkers()


@app.post("/api/chunk")
async def api_chunk_document(
    file,
    converter_config: ConverterConfig,
    chunker_config: ChunkerConfig,
    pdf_password: str | None = None,
):
    """Convert and chunk a document."""
    return await routes.chunk_document(
        file, converter_config, chunker_config, pdf_password
    )


@app.post("/api/pdf/metadata")
async def api_get_pdf_metadata(
    file: UploadFile = File(..., description="The PDF file to analyze"),  # noqa: B008
    pdf_password: str | None = Form(None, description="Password for encrypted PDF files"),
):
    """Get PDF metadata including page count and document information."""
    return await routes.get_pdf_metadata(file, pdf_password)


# Additional endpoints for monitoring
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
