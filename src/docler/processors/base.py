"""Base class for document processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.configs.processor_configs import BaseProcessorConfig
    from docler.models import Document


class DocumentProcessor[TConfig](BaseProvider[TConfig], ABC):
    """Base class for document pre-processors."""

    Config: ClassVar[type[BaseProcessorConfig]]
    """Configuration class for this processor."""

    @abstractmethod
    async def process(self, doc: Document) -> Document:
        """Process a document to improve its content.

        Args:
            doc: Document to process

        Returns:
            Processed document with improved content
        """
        raise NotImplementedError
