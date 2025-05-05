"""Configuration models for document and image annotators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field
from schemez import ModelIdentifier  # noqa: TC002

from docler.common_types import DEFAULT_ANNOTATOR_MODEL, DEFAULT_IMAGE_ANNOTATOR_MODEL
from docler.provider import ProviderConfig


if TYPE_CHECKING:
    from docler.annotators.base import Annotator


DEFAULT_DOC_SYSTEM_PROMPT = """
You are an expert document analyzer that extracts meaningful metadata.
For each document or text chunk, extract:
1. Main topics (3-5 categories)
2. Key entities (people, organizations, locations, products)
3. Keywords (5-10 important terms)

Format your response as structured data that can be parsed as JSON.
"""

DEFAULT_DOC_PROMPT_TEMPLATE = """
Complete context:
    {context}

Please analyze and describe this text chunk:
    {chunk}
"""

DEFAULT_IMAGE_SYSTEM_PROMPT = """
Analyze images in detail. For each image, provide:
1. A detailed description of what's visible
2. Key objects/people present
3. Any text content visible in the image
4. Image type (photo, chart, diagram, illustration, etc.)
5. Dominant colors and visual elements

Format your response as structured data that can be parsed as JSON.
"""

DEFAULT_IMAGE_PROMPT_TEMPLATE = """
Analyze this image with ID {image_id}{filename_info}.
Describe what you see and extract key information.
"""


class BaseAnnotatorConfig(ProviderConfig):
    """Base configuration for annotators."""


class AIDocumentAnnotatorConfig(BaseAnnotatorConfig):
    """Configuration for AI-based document and chunk annotator."""

    type: Literal["ai_document"] = Field(default="ai_document", init=False)
    """Type discriminator for AI document annotator."""

    model: ModelIdentifier = DEFAULT_ANNOTATOR_MODEL
    """LLM model to use for annotation."""

    system_prompt: str = DEFAULT_DOC_SYSTEM_PROMPT
    """System prompt for the annotation task."""

    user_prompt: str = DEFAULT_DOC_PROMPT_TEMPLATE
    """Template for the annotation prompt."""

    max_context_length: int = Field(default=1500, ge=1)
    """Maximum length of context for annotation."""

    batch_size: int = Field(default=5, ge=1)
    """Number of chunks to process in parallel."""

    def get_provider(self) -> Annotator:
        """Get the annotator instance."""
        from docler.annotators.ai_document_annotator import AIDocumentAnnotator

        return AIDocumentAnnotator(**self.get_config_fields())


class AIImageAnnotatorConfig(BaseAnnotatorConfig):
    """Configuration for AI-based image annotator."""

    type: Literal["ai_image"] = Field(default="ai_image", init=False)
    """Type discriminator for AI image annotator."""

    model: ModelIdentifier = DEFAULT_IMAGE_ANNOTATOR_MODEL
    """Vision model to use for image annotation."""

    system_prompt: str = DEFAULT_IMAGE_SYSTEM_PROMPT
    """System prompt for the image analysis task."""

    user_prompt: str = DEFAULT_IMAGE_PROMPT_TEMPLATE
    """User prompt template for image analysis."""

    batch_size: int = Field(default=3, ge=1)
    """Number of images to process concurrently."""

    def get_provider(self) -> Annotator:
        """Get the annotator instance."""
        from docler.annotators.ai_image_annotator import AIImageAnnotator

        return AIImageAnnotator(**self.get_config_fields())


# Union type for annotator configs
AnnotatorConfig = Annotated[
    AIDocumentAnnotatorConfig | AIImageAnnotatorConfig,
    Field(discriminator="type"),
]
