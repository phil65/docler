"""API route implementations for Docler."""

from __future__ import annotations

import mimetypes
import tempfile
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Body, File, Form, HTTPException, Query, UploadFile
from pydantic import TypeAdapter
import upath

from docler.configs.converter_configs import ConverterConfig
from docler.models import PageMetadata  # noqa: TC001
from docler.pdf_utils import get_pdf_info


if TYPE_CHECKING:
    from docler.configs.chunker_configs import ChunkerConfig


config_adapter = TypeAdapter[ConverterConfig](ConverterConfig)


async def convert_document(
    file: Annotated[UploadFile, File(description="The document file to convert")],
    config: Annotated[str, Form(description="Converter configuration JSON")],
    include_images_as_base64: Annotated[
        bool, Form(description="Whether to include image data as base64 in the response")
    ] = True,
):
    """Convert a document file to markdown using specified converter configuration."""
    # Parse the JSON config string manually
    import anyenv

    try:
        config_dict = anyenv.load_json(config)
        parsed = config_adapter.validate_python(config_dict)
    except (anyenv.JsonLoadError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")  # noqa: B904

    content = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=f"_{file.filename}", delete=False
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(content)

    try:
        mime_type, _ = mimetypes.guess_type(file.filename or "")
        if not mime_type:
            mime_type = file.content_type

        # Create converter from config
        converter = parsed.get_provider()
        document = await converter.convert_file(temp_path)

        # Process images based on the flag
        if include_images_as_base64:
            # Ensure images are in base64 format
            for image in document.images:
                if isinstance(image.content, bytes):
                    # Convert bytes to base64 string
                    image.content = image.to_base64()
        else:
            # Remove binary content from images
            for image in document.images:
                image.content = ""
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(
                status_code=500, detail=f"Error during document conversion: {e!s}"
            ) from None
        raise
    else:
        return document
    finally:
        # Clean up the temporary file
        path = upath.UPath(temp_path)
        path.unlink(missing_ok=True)


async def chunk_document(
    file: Annotated[UploadFile, File(description="The document file to chunk")],
    converter_config: Annotated[
        ConverterConfig,
        Body(default={"type": "marker"}, description="Converter configuration"),
    ],
    chunker_config: Annotated[
        ChunkerConfig,
        Body(default={"type": "markdown"}, description="Chunker configuration"),
    ],
    include_images_as_base64: bool = Query(
        default=True,
        description="Whether to include image data as base64 in the response",
    ),
):
    """Convert and chunk a document file using specified configurations.

    Args:
        file: The document file to convert and chunk
        converter_config: Configuration for the document converter
        chunker_config: Configuration for the text chunker
        include_images_as_base64: Whether to include image data as base64 in the response

    Returns:
        JSON response with the chunked document
    """
    content = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=f"_{file.filename}", delete=False
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(content)

    try:
        mime_type, _ = mimetypes.guess_type(file.filename or "")
        if not mime_type:
            mime_type = file.content_type

        # Get converter and chunker instances
        converter = converter_config.get_provider()
        chunker = chunker_config.get_provider()

        # Convert the document
        document = await converter.convert_file(temp_path)

        # Chunk the document
        chunked_document = await chunker.chunk(document)

        # Process images based on the flag
        if include_images_as_base64:
            # Ensure images are in base64 format
            for image in chunked_document.images:
                if isinstance(image.content, bytes):
                    # Convert bytes to base64 string
                    image.content = image.to_base64()
            for chunk in chunked_document.chunks:
                for image in chunk.images:
                    if isinstance(image.content, bytes):
                        # Convert bytes to base64 string
                        image.content = image.to_base64()
        else:
            # Remove binary content from images
            for image in chunked_document.images:
                image.content = ""
            for chunk in chunked_document.chunks:
                for image in chunk.images:
                    image.content = ""
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(
                status_code=500,
                detail=f"Error during document conversion or chunking: {e!s}",
            ) from e
        raise
    else:
        return chunked_document
    finally:
        # Clean up the temporary file
        path = upath.UPath(temp_path)
        path.unlink(missing_ok=True)


async def list_converters():
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


async def list_chunkers():
    """List all available chunkers."""
    from docler.chunkers.base import TextChunker

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
) -> PageMetadata:
    """Get PDF metadata including page count and document information.

    Args:
        file: The PDF file to analyze

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
        raise HTTPException(
            status_code=400, detail="File size too large. Maximum size is 100MB."
        )

    try:
        content = await file.read()
        metadata = get_pdf_info(content)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process PDF: {e!s}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal error processing PDF: {e!s}"
        ) from e
    else:
        return metadata
