"""API route implementations for Docler."""

from __future__ import annotations

import mimetypes
import tempfile
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Body, File, HTTPException, Query, UploadFile
import upath


if TYPE_CHECKING:
    from docler.configs.chunker_configs import ChunkerConfig
    from docler.configs.converter_configs import ConverterConfig


async def convert_document(
    file: Annotated[UploadFile, File(description="The document file to convert")],
    config: Annotated[
        ConverterConfig,
        Body(default={"type": "marker"}, description="Converter configuration"),
    ],
    include_images_as_base64: bool = Query(
        default=True,
        description="Whether to include image data as base64 in the response",
    ),
):
    """Convert a document file to markdown using specified converter configuration.

    Args:
        file: The document file to convert (PDF, DOCX, images, etc.)
        config: Configuration for the converter (uses docler's configuration models)
        include_images_as_base64: Whether to include image data as base64 in the response

    Returns:
        JSON response with the converted document
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

        # Create converter from config
        converter = config.get_provider()
        document = await converter.convert_file(temp_path)

        # Process images if requested
        if not include_images_as_base64:
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

        # Process images if requested
        if not include_images_as_base64:
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
        }
        for converter in registry._converters
    ]

    return {"converters": converters}


async def list_chunkers():
    """List all available chunkers."""
    from docler.chunkers.base import TextChunker

    chunker_classes = TextChunker[Any].get_available_providers()
    chunkers = [
        {"name": chunker_class.NAME, "config_type": chunker_class.Config.__name__}
        for chunker_class in chunker_classes
        if chunker_class.has_required_packages()
    ]

    return {"chunkers": chunkers}
