"""FastAPI server for docler document conversion library."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from docler import __version__ as docler_version
from docler.models import PageMetadata  # noqa: TC001
from docler_api import routes
from docler_api.guard import ApiGuard
from docler_api.models import ConvertResponse, ImagesMode  # noqa: TC001
from docler_api.settings import ApiSettings


v1 = APIRouter(prefix="/v1")


@v1.post("/convert")
async def api_convert(
    file: Annotated[UploadFile, File(description="The document file to convert")],
    converter: Annotated[str, Form(description="Converter configuration JSON")],
    chunker: Annotated[
        str | None, Form(description="Chunker configuration JSON; chunks the document")
    ] = None,
    pdf_password: Annotated[
        str | None, Form(description="Password for encrypted PDF files")
    ] = None,
    images: Annotated[
        ImagesMode, Form(description="`inline`: base64, each image once; `none`: no image data")
    ] = "inline",
) -> ConvertResponse:
    """Convert a document to markdown, and chunk it when a chunker config is given."""
    return await routes.convert(file, converter, chunker, pdf_password, images)


@v1.post("/pdf/metadata")
async def api_pdf_metadata(
    file: Annotated[UploadFile, File(description="The PDF file to analyze")],
    pdf_password: Annotated[
        str | None, Form(description="Password for encrypted PDF files")
    ] = None,
) -> PageMetadata:
    """Get PDF metadata including page count and document information."""
    return await routes.get_pdf_metadata(file, pdf_password)


@v1.get("/converters")
async def api_converters() -> dict[str, list[dict[str, Any]]]:
    """List all available converters."""
    return await routes.list_converters()


@v1.get("/chunkers")
async def api_chunkers() -> dict[str, list[dict[str, Any]]]:
    """List all available chunkers."""
    return await routes.list_chunkers()


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create the API app.

    Args:
        settings: Server settings; read from the environment (`DOCLER_API_*`) if None.
    """
    settings = settings or ApiSettings.from_env()
    app = FastAPI(
        title="Docler API",
        description="API for document conversion using docler",
        version=docler_version,
    )
    app.include_router(v1)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/ready")
    async def readiness_check() -> dict[str, str]:
        """Readiness check endpoint."""
        return {"status": "ready"}

    app.add_middleware(ApiGuard, settings=settings)
    if settings.cors_origins:
        # Added last, so it wraps the guard: preflights and error responses get CORS
        # headers. The token travels in a header, so no credentials mode.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
