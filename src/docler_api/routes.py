"""API route implementations for Docler."""

import mimetypes
from typing import TYPE_CHECKING, Any

import anyenv
from fastapi import HTTPException
from pydantic import TypeAdapter

from docler.pdf_utils import get_pdf_info
from docler_api.models import ConvertResponse, ImagesMode, build_response
from docler_config.chunker_configs import ChunkerConfig
from docler_config.converter_configs import ConverterConfig


if TYPE_CHECKING:
    from fastapi import UploadFile

    from docler.models import PageMetadata


config_adapter = TypeAdapter[ConverterConfig](ConverterConfig)
chunker_config_adapter = TypeAdapter[ChunkerConfig](ChunkerConfig)

# Fields that pick the host a provider sends the server's API keys (from env) and the
# uploaded file to. Clients must not set them: the server configures them via env
# (UPSTAGE_BASE_URL, AZURE_DOC_INTELLIGENCE_ENDPOINT, DOCLING_REMOTE_ENDPOINT).
# Keys are compared lowercased: pydantic-settings matches field names case-insensitively.
SERVER_ONLY_FIELDS = frozenset({"base_url", "endpoint"})


def _find_server_only_fields(value: Any, path: str = "") -> list[str]:
    match value:
        case dict():
            found = [f"{path}{k}" for k in value if str(k).lower() in SERVER_ONLY_FIELDS]
            for k, v in value.items():
                found.extend(_find_server_only_fields(v, f"{path}{k}."))
            return found
        case list():
            return [
                f for i, v in enumerate(value) for f in _find_server_only_fields(v, f"{path}{i}.")
            ]
        case _:
            return []


def parse_client_config[T](adapter: TypeAdapter[T], config: str) -> T:
    """Parse a config sent by a client, rejecting fields only the server may set.

    Raises:
        HTTPException: 400 if the JSON is invalid or sets a server-only field.
    """
    try:
        config_dict = anyenv.load_json(config)
    except anyenv.JsonLoadError as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}") from e
    if forbidden := _find_server_only_fields(config_dict):
        detail = f"Config fields can only be set on the server: {', '.join(forbidden)}"
        raise HTTPException(status_code=400, detail=detail)
    try:
        return adapter.validate_python(config_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}") from e


def _mime_type(file: UploadFile) -> str:
    mime_type, _ = mimetypes.guess_type(file.filename or "")
    return mime_type or file.content_type or "application/octet-stream"


async def convert(
    file: UploadFile,
    converter: str,
    chunker: str | None = None,
    pdf_password: str | None = None,
    images: ImagesMode = "inline",
) -> ConvertResponse:
    """Convert a document and, with a chunker config, chunk it.

    Args:
        file: The document file.
        converter: Converter configuration JSON.
        chunker: Chunker configuration JSON (optional).
        pdf_password: Password for encrypted PDF files.
        images: `inline` (base64, each image once) or `none` (ids and metadata only).

    Raises:
        HTTPException: 400 for a bad config or document, 401 for a wrong PDF password,
            500 for other conversion errors.
    """
    converter_config = parse_client_config(config_adapter, converter)
    chunker_config = parse_client_config(chunker_config_adapter, chunker) if chunker else None
    content = await file.read()
    try:
        document = await converter_config.get_provider().convert_content(
            content=content,
            mime_type=_mime_type(file),
            source_path=file.filename,
            title=file.filename,
            pdf_password=pdf_password,
        )
        chunks = None
        if chunker_config:
            chunks = (await chunker_config.get_provider().chunk(document)).chunks
    except HTTPException:
        raise
    except ValueError as e:
        if "password" in str(e).lower():
            status_code = 401 if "incorrect" in str(e).lower() else 400
            raise HTTPException(status_code=status_code, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=f"Error during conversion: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during conversion: {e!s}") from e
    return build_response(document, chunks, images)


async def list_converters() -> dict[str, list[dict[str, Any]]]:
    """List all available converters."""
    from docler.converters.registry import ConverterRegistry

    registry = ConverterRegistry.create_default()

    converters = [
        {
            "name": converter.NAME,
            "supported_mime_types": list(converter.SUPPORTED_MIME_TYPES),
            "config_type": converter.__class__.Config.__name__,
            "config_schema": converter.__class__.Config.model_json_schema(),
        }
        for converter in registry._converters
    ]

    return {"converters": converters}


async def list_chunkers() -> dict[str, list[dict[str, Any]]]:
    """List all available chunkers."""
    # Chunkers register on import; docler.chunkers itself imports none of them.
    import docler.chunkers.ai_chunker
    from docler.chunkers.base import TextChunker
    import docler.chunkers.llamaindex_chunker
    import docler.chunkers.markdown_chunker
    import docler.chunkers.token_chunker  # noqa: F401

    chunker_classes = TextChunker[Any].get_available_providers()
    chunkers = [
        {
            "name": chunker_class.NAME,
            "config_type": chunker_class.Config.__name__,
            "config_schema": chunker_class.Config.model_json_schema(),
        }
        for chunker_class in chunker_classes
        if chunker_class.has_required_packages()
    ]

    return {"chunkers": chunkers}


async def get_pdf_metadata(file: UploadFile, pdf_password: str | None = None) -> PageMetadata:
    """Get PDF metadata including page count and document information.

    Args:
        file: The PDF file to analyze
        pdf_password: Password for encrypted PDF files

    Returns:
        PageMetadata containing document information

    Raises:
        HTTPException: If file is invalid or processing fails
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        content = await file.read()
        metadata = get_pdf_info(content, pdf_password)
    except ValueError as e:
        if "Incorrect password" in str(e):
            raise HTTPException(status_code=401, detail="Incorrect PDF password") from e
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error processing PDF: {e!s}") from e
    else:
        return metadata
