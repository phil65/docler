"""AI-powered document and chunk metadata annotation."""

from __future__ import annotations

import asyncio
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pydantic import BaseModel

from docler.annotators.base import Annotator
from docler.common_types import DEFAULT_ANNOTATOR_MODEL
from docler.configs.annotator_configs import (
    DEFAULT_DOC_PROMPT_TEMPLATE,
    DEFAULT_DOC_SYSTEM_PROMPT,
    AIDocumentAnnotatorConfig,
)
from docler.log import get_logger


if TYPE_CHECKING:
    from docler.models import ChunkedDocument

logger = get_logger(__name__)


class DefaultMetadata(BaseModel):
    """Default metadata for a document or chunk."""

    topics: list[str]
    """Topics/categories."""

    keywords: list[str]
    """Keywords."""

    entities: list[str]
    """Main entities."""


# Type variable for generic metadata model
T = TypeVar("T", bound=BaseModel)


class AIDocumentAnnotator[TMetadata](Annotator[AIDocumentAnnotatorConfig]):
    """AI-based document and chunk annotator.

    Enhances documents and chunks with metadata.

    Type Parameters:
        T: Type of metadata model to use. Must be a Pydantic BaseModel.
    """

    Config = AIDocumentAnnotatorConfig
    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        metadata_model: type[TMetadata] = DefaultMetadata,  # type: ignore
        max_context_length: int = 1500,
        batch_size: int = 5,
    ):
        """Initialize the AI document annotator.

        Args:
            model: LLM model to use for annotation
            system_prompt: Optional custom prompt for annotation
            user_prompt: Optional custom prompt for annotation
            metadata_model: Pydantic model class for metadata structure
            max_context_length: Maximum length of context for annotation
            batch_size: Number of chunks to process in parallel
        """
        self.model = model or DEFAULT_ANNOTATOR_MODEL
        self.system_prompt = system_prompt or DEFAULT_DOC_SYSTEM_PROMPT
        self.user_prompt = user_prompt or DEFAULT_DOC_PROMPT_TEMPLATE
        self.metadata_model = metadata_model
        self.max_context_length = max_context_length
        self.batch_size = batch_size

    async def annotate(self, document: ChunkedDocument) -> ChunkedDocument:
        """Annotate document and chunks with AI-generated metadata.

        Args:
            document: Chunked document to annotate

        Returns:
            Document with enhanced metadata
        """
        from llmling_agent import Agent, StructuredAgent

        agent: StructuredAgent[None, TMetadata] = Agent[None](  # type: ignore
            model=self.model,
            system_prompt=self.system_prompt,
        ).to_structured(self.metadata_model)

        # Process document-level metadata if needed
        if document.metadata is None:
            document.metadata = {}

        # Get a condensed version of the document for context
        context = (
            document.content[: self.max_context_length] + "..."
            if self.max_context_length and len(document.content) > self.max_context_length
            else document.content
        )

        # Process chunks in batches
        for batch in batched(document.chunks, self.batch_size):
            tasks = []

            for chunk in batch:
                prompt = self.user_prompt.format(context=context, chunk=chunk.text)
                tasks.append(agent.run(prompt))

            try:
                results = await asyncio.gather(*tasks)
                for chunk, result in zip(batch, results):
                    metadata = result.content.model_dump()  # type: ignore
                    chunk.metadata |= metadata
            except Exception:
                logger.exception("Error annotating batch")

        return document
