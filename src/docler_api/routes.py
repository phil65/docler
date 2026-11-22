"""API route implementations for Docler."""

from __future__ import annotations

import mimetypes
from typing import TYPE_CHECKING, Annotated, Any

import anyenv
from fastapi import File, Form, HTTPException
from pydantic import TypeAdapter

from docler.models import ChunkedDocument, PageMetadata  # noqa: TC001
from docler.pdf_utils import get_pdf_info
from docler_config.chunker_configs import ChunkerConfig
from docler_config.converter_configs import ConverterConfig


if TYPE_CHECKING:
    from fastapi import UploadFile
    from mkdown import Document


config_adapter = TypeAdapter[ConverterConfig](ConverterConfig)
chunker_config_adapter = TypeAdapter[ChunkerConfig](ChunkerConfig)


async def convert_document(
    file: Annotated[UploadFile, File(description="The document file to convert")],
    config: Annotated[str, Form(description="Converter configuration JSON")],
    pdf_password: Annotated[
        str | None, Form(description="Password for encrypted PDF files")
    ] = None,
) -> Document:
    """Convert a document file to markdown using specified converter configuration."""
    # Parse the JSON config string manually

    try:
        config_dict = anyenv.load_json(config)
        parsed = config_adapter.validate_python(config_dict)
    except (anyenv.JsonLoadError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")  # noqa: B904

    content = await file.read()

    try:
        mime_type, _ = mimetypes.guess_type(file.filename or "")
        if not mime_type:
            mime_type = file.content_type or "application/octet-stream"

        # Create converter from config
        converter = parsed.get_provider()
        document = await converter.convert_content(
            content=content,
            mime_type=mime_type,
            source_path=file.filename,
            title=file.filename,
            pdf_password=pdf_password,
        )

        # Always include images as base64
        for image in document.images:
            if isinstance(image.content, bytes):
                # Convert bytes to base64 string
                image.content = image.to_base64()
    except ValueError as e:
        if "password" in str(e).lower():
            status_code = 401 if "incorrect" in str(e).lower() else 400
            raise HTTPException(status_code=status_code, detail=str(e)) from e
        raise HTTPException(
            status_code=400, detail=f"Error during document conversion: {e!s}"
        ) from e
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(
                status_code=500, detail=f"Error during document conversion: {e!s}"
            ) from None
        raise
    else:
        return document


async def chunk_document(
    file: Annotated[UploadFile, File(description="The document file to chunk")],
    converter_config: Annotated[
        str, Form(description="Converter configuration JSON")
    ] = '{"type": "marker"}',
    chunker_config: Annotated[
        str, Form(description="Chunker configuration JSON")
    ] = '{"type": "markdown"}',
    pdf_password: Annotated[
        str | None, Form(description="Password for encrypted PDF files")
    ] = None,
) -> ChunkedDocument:
    """Convert and chunk a document file using specified configurations.

    Args:
        file: The document file to convert and chunk
        converter_config: Converter configuration as JSON string
        chunker_config: Chunker configuration as JSON string
        pdf_password: Password for encrypted PDF files

    Returns:
        JSON response with the chunked document
    """
    try:
        converter = config_adapter.validate_python(anyenv.load_json(converter_config))
        chunker = chunker_config_adapter.validate_python(anyenv.load_json(chunker_config))
    except (anyenv.JsonLoadError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")  # noqa: B904

    content = await file.read()

    try:
        mime_type, _ = mimetypes.guess_type(file.filename or "")
        if not mime_type:
            mime_type = file.content_type or "application/octet-stream"
        document = await converter.get_provider().convert_content(
            content=content,
            mime_type=mime_type,
            source_path=file.filename,
            title=file.filename,
            pdf_password=pdf_password,
        )

        # Chunk the document
        chunked_document = await chunker.get_provider().chunk(document)

        # Always include images as base64
        for image in chunked_document.images:
            if isinstance(image.content, bytes):
                # Convert bytes to base64 string
                image.content = image.to_base64()
        for chunk in chunked_document.chunks:
            for image in chunk.images:
                if isinstance(image.content, bytes):
                    # Convert bytes to base64 string
                    image.content = image.to_base64()
    except ValueError as e:
        if "password" in str(e).lower():
            status_code = 401 if "incorrect" in str(e).lower() else 400
            raise HTTPException(status_code=status_code, detail=str(e)) from e
        raise HTTPException(
            status_code=400, detail=f"Error during document conversion or chunking: {e!s}"
        ) from e
    except Exception as e:
        if not isinstance(e, HTTPException):
            msg = f"Error during document conversion or chunking: {e!s}"
            raise HTTPException(status_code=500, detail=msg) from e
        raise
    else:
        return chunked_document


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


async def get_pdf_metadata(
    file: Annotated[UploadFile, File(description="The PDF file to analyze")],
    pdf_password: Annotated[
        str | None, Form(description="Password for encrypted PDF files")
    ] = None,
) -> PageMetadata:
    """Get PDF metadata including page count and document information.

    Args:
        file: The PDF file to analyze
        pdf_password: Password for encrypted PDF files

    Returns:
        PageMetadata containing document information

    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate file size (100MB limit for metadata extraction)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large. Maximum size is 100MB.")

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
