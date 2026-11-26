"""Base classes for text chunking implementations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.models import ChunkedDocument
    from docler_config.annotator_configs import BaseAnnotatorConfig


class Annotator[TConfig: BaseModel](ABC, BaseProvider[TConfig]):
    """Base class for chunk annotation processors."""

    Config: ClassVar[type[BaseAnnotatorConfig]]

    @abstractmethod
    async def annotate(
        self,
        document: ChunkedDocument,
    ) -> ChunkedDocument:
        """Annotate a chunked document with additional metadata.

        Args:
            document: Chunked document containing the original content and chunks

        Returns:
            Chunked document with annotated chunks
        """
        raise NotImplementedError
