"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.models import ChunkedDocument


class Annotator(ABC):
    """Base class for chunk annotation processors."""

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
