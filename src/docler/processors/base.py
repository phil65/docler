from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.models import Document


class DocumentProcessor(ABC):
    """Base class for document pre-processors."""

    @abstractmethod
    async def process(self, doc: Document) -> Document:
        """Process a document to improve its content.

        Args:
            doc: Document to process

        Returns:
            Processed document with improved content
        """
        raise NotImplementedError
