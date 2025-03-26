"""AI-powered image annotation."""

from __future__ import annotations

import asyncio
import base64
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pydantic import BaseModel

from docler.annotators.base import Annotator
from docler.common_types import DEFAULT_IMAGE_ANNOTATOR_MODEL
from docler.log import get_logger


if TYPE_CHECKING:
    from docler.models import ChunkedDocument, Image

logger = get_logger(__name__)


class DefaultImageMetadata(BaseModel):
    """Default metadata for an image."""

    description: str
    """Detailed description of the image."""

    objects: list[str]
    """Objects identified in the image."""

    text_content: str | None = None
    """Text visible in the image, if any."""

    image_type: str
    """Type of image (photo, diagram, chart, etc.)."""

    colors: list[str]
    """Dominant colors in the image."""


# Type variable for generic metadata model
TMetadata = TypeVar("TMetadata", bound=BaseModel)


SYSTEM_PROMPT = """
Analyze images in detail. For each image, provide:
1. A detailed description of what's visible
2. Key objects/people present
3. Any text content visible in the image
4. Image type (photo, chart, diagram, illustration, etc.)
5. Dominant colors and visual elements

Format your response as structured data that can be parsed as JSON.
"""

USER_PROMPT = """
Analyze this image with ID {image_id}{filename_info}.
Describe what you see and extract key information.
"""


class AIImageAnnotator[TMetadata](Annotator):
    """AI-based image annotator.

    Analyzes images in chunks and adds descriptions and metadata.

    Type Parameters:
        TMetadata: Type of metadata model to use. Must be a Pydantic BaseModel.
    """

    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        metadata_model: type[TMetadata] = DefaultImageMetadata,  # type: ignore
        batch_size: int = 3,
    ):
        """Initialize the AI image annotator.

        Args:
            model: Vision model to use (must support images)
            system_prompt: Custom prompt for image analysis
            user_prompt: Custom user prompt template for image analysis
            metadata_model: Pydantic model for image metadata
            batch_size: Number of images to process concurrently
        """
        self.model = model or DEFAULT_IMAGE_ANNOTATOR_MODEL
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.user_prompt = user_prompt or USER_PROMPT
        self.metadata_model = metadata_model
        self.batch_size = batch_size

    async def _process_image(self, image: Image) -> Image:
        """Process a single image with the vision model.

        Args:
            image: Image to analyze

        Returns:
            Image with added description and metadata
        """
        from llmling_agent import Agent, ImageBase64Content

        if image.description and image.metadata:
            return image

        if isinstance(image.content, bytes):
            b64_content = base64.b64encode(image.content).decode("utf-8")
        else:
            b64_content = (
                image.content
                if not image.content.startswith("data:")
                else image.content.split(",", 1)[1]
            )
        img_content = ImageBase64Content(data=b64_content, mime_type=image.mime_type)
        agent = Agent[None](
            model=self.model,
            system_prompt=self.system_prompt,
        ).to_structured(self.metadata_model)

        try:
            filename_info = f" ({image.filename})" if image.filename else ""
            prompt = self.user_prompt.format(
                image_id=image.id,
                filename_info=filename_info,
                filename=image.filename or "",
                mime_type=image.mime_type,
            )

            result = await agent.run(prompt, img_content)
            metadata = result.content.model_dump()  # pyright: ignore
            description = metadata.pop("description", None)
            if description:
                image.description = description
            if image.metadata is None:
                image.metadata = {}
            image.metadata.update(metadata)

        except Exception:
            logger.exception("Error processing image %s", image.id)

        return image

    async def annotate(self, document: ChunkedDocument) -> ChunkedDocument:
        """Annotate all images in the chunks with AI-generated descriptions.

        Args:
            document: Chunked document containing chunks with images to annotate

        Returns:
            Document with annotated images in chunks
        """
        # Process images in chunks only
        for chunk in document.chunks:
            if not chunk.images:
                continue

            # Process images in this chunk
            for batch in batched(chunk.images, self.batch_size):
                tasks = [self._process_image(img) for img in batch]
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    logger.exception(
                        "Error processing images in chunk %s", chunk.chunk_index
                    )

        return document
