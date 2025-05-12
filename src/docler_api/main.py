"""FastAPI server for docler document conversion library."""

from __future__ import annotations

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
    file: UploadFile,
    config: ConverterConfig,
    include_images_as_base64: bool = True,
):
    """Convert a document file to markdown."""
    return await routes.convert_document(file, config, include_images_as_base64)


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
    include_images_as_base64: bool = True,
):
    """Convert and chunk a document."""
    return await routes.chunk_document(
        file, converter_config, chunker_config, include_images_as_base64
    )


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
