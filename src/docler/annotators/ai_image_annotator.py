"""AI-powered image annotation."""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING, ClassVar

import anyenv
from schemez import Schema
from typing_extensions import TypeVar

from docler.annotators.base import Annotator
from docler.common_types import DEFAULT_IMAGE_ANNOTATOR_MODEL
from docler.configs.annotator_configs import (
    DEFAULT_IMAGE_PROMPT_TEMPLATE,
    DEFAULT_IMAGE_SYSTEM_PROMPT,
    AIImageAnnotatorConfig,
)


if TYPE_CHECKING:
    from mkdown import Image

    from docler.models import ChunkedDocument


class DefaultImageMetadata(Schema):
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


TMetadata = TypeVar("TMetadata", bound=Schema, default=DefaultImageMetadata)


class AIImageAnnotator[TMetadata](Annotator[AIImageAnnotatorConfig]):
    """AI-based image annotator.

    Analyzes images in chunks and adds descriptions and metadata.

    Type Parameters:
        TMetadata: Type of metadata model to use. Must be a Pydantic Schema.
    """

    Config = AIImageAnnotatorConfig

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
        super().__init__()
        self.model = model or DEFAULT_IMAGE_ANNOTATOR_MODEL
        self.system_prompt = system_prompt or DEFAULT_IMAGE_SYSTEM_PROMPT
        self.user_prompt = user_prompt or DEFAULT_IMAGE_PROMPT_TEMPLATE
        self.metadata_model = metadata_model
        self.batch_size = batch_size

    async def _process_image(self, image: Image) -> Image:
        """Process a single image with the vision model.

        Args:
            image: Image to analyze

        Returns:
            Image with added description and metadata
        """
        from llmling_agent import Agent, ImageBase64Content, StructuredAgent

        if image.description and image.metadata:
            return image

        b64_content = image.to_base64()
        img_content = ImageBase64Content(data=b64_content, mime_type=image.mime_type)
        agent: StructuredAgent[None, TMetadata] = Agent[None](  # type: ignore
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
            metadata = result.content.model_dump()  # type: ignore
            description = metadata.pop("description", None)
            if description:
                image.description = description
            image.metadata.update(metadata)

        except Exception:
            self.logger.exception("Error processing image %s", image.id)

        return image

    async def annotate(self, document: ChunkedDocument) -> ChunkedDocument:
        """Annotate all images in the chunks with AI-generated descriptions.

        Args:
            document: Chunked document containing chunks with images to annotate

        Returns:
            Document with annotated images in chunks
        """
        for chunk in document.chunks:
            if not chunk.images:
                continue

            for batch in batched(chunk.images, self.batch_size):
                tasks = [self._process_image(img) for img in batch]
                try:
                    await anyenv.gather(*tasks)
                except Exception:
                    msg = "Error processing images in chunk %s"
                    self.logger.exception(msg, chunk.chunk_index)

        return document


if __name__ == "__main__":
    import asyncio

    from mkdown import Image, TextChunk

    from docler.models import ChunkedDocument

    async def main():
        annotator = AIImageAnnotator[DefaultImageMetadata]()
        url = "https://www.a-i-stack.com/wp-content/uploads/go-x/u/93dcedb9-17f3-4aee-9b5a-3744e5e84686/image-342x342.png"
        image = await Image.from_file(url)
        chunk = TextChunk("Sample text", "sample_doc_id", images=[image], chunk_index=0)
        document = ChunkedDocument(content="test", chunks=[chunk])
        doc = await annotator.annotate(document)
        print(doc)

    asyncio.run(main())
