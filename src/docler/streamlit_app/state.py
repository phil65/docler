"""Session state management for the Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from streambricks import State

from docler.models import ChunkedDocument, Document  # noqa: TC001


if TYPE_CHECKING:
    from docler.configs.vector_db_configs import (
        ChromaConfig,
        OpenAIVectorConfig,
        PineconeConfig,
        QdrantConfig,
    )


class SessionState(State):
    step: int = 1
    document: Document | None = None
    uploaded_file_name: str | None = None
    chunked_doc: ChunkedDocument | None = None
    vector_store_id: str | None = None
    vector_provider: str | None = None
    uploaded_chunks: int | None = None
    chunks: list[str] | None = None
    vector_configs: dict[
        str, ChromaConfig | OpenAIVectorConfig | PineconeConfig | QdrantConfig
    ] = Field(default_factory=dict)

    def next_step(self):
        """Move to the next step in the workflow."""
        self.step += 1

    def prev_step(self):
        """Move to the previous step in the workflow."""
        self.step -= 1
