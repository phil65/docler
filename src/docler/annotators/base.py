"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.configs.annotator_configs import BaseAnnotatorConfig
    from docler.models import ChunkedDocument


class Annotator[TConfig](ABC, BaseProvider[TConfig]):
    """Base class for chunk annotation processors."""

    Config: ClassVar[type[BaseAnnotatorConfig]]

    @abstractmethod
    async def annotate(
        self,
        chunked_doc: ChunkedDocument,
    ) -> ChunkedDocument:
        """Annotate a chunked document with additional metadata.

        Args:
            chunked_doc: Chunked document containing the original content and chunks

        Returns:
            Chunked document with annotated chunks
        """
        raise NotImplementedError
