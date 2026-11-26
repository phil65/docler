"""Response models of the Docler API."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from docler.markdown import Document, Image, TextChunk


ImagesMode = Literal["inline", "none"]


class ImageOut(BaseModel):
    """An image of the document; markdown references it by `filename` (or `id`)."""

    id: str
    mime_type: str
    filename: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str | None = None
    """Base64 image data; None with `images=none`."""


class DocumentOut(BaseModel):
    """The converted document, without its images (they are in the response's `images`)."""

    content: str
    title: str | None = None
    author: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
    source_path: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkOut(BaseModel):
    """A chunk of the document; its images are in the response's `images`."""

    content: str
    chunk_index: int
    page_number: int | None = None
    image_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConvertResponse(BaseModel):
    """Response of `/v1/convert`: each image once, chunks only when a chunker was given."""

    document: DocumentOut
    images: list[ImageOut] = Field(default_factory=list)
    chunks: list[ChunkOut] | None = None


def _image_out(image: Image, mode: ImagesMode) -> ImageOut:
    return ImageOut(
        id=image.id,
        mime_type=image.mime_type,
        filename=image.filename,
        description=image.description,
        metadata=image.metadata,
        content=image.to_base64() if mode == "inline" else None,
    )


def build_response(
    document: Document, chunks: list[TextChunk] | None, images: ImagesMode
) -> ConvertResponse:
    """Assemble the response, listing every image once (by id)."""
    by_id = {image.id: image for image in document.images}
    for chunk in chunks or []:
        for image in chunk.images:
            by_id.setdefault(image.id, image)
    return ConvertResponse(
        document=DocumentOut(
            content=document.content,
            title=document.title,
            author=document.author,
            created=document.created,
            modified=document.modified,
            source_path=document.source_path,
            mime_type=document.mime_type,
            metadata=document.metadata,
        ),
        images=[_image_out(image, images) for image in by_id.values()],
        chunks=None
        if chunks is None
        else [
            ChunkOut(
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                image_ids=[image.id for image in chunk.images],
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ],
    )
