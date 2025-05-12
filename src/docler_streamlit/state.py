"""Session state management for the Streamlit app."""

from __future__ import annotations

from mkdown import Document  # noqa: TC002
from pydantic import Field
from streambricks import State

from docler.configs.converter_configs import BaseConverterConfig  # noqa: TC001
from docler.configs.processor_configs import LLMProofReaderConfig  # noqa: TC001
from docler.configs.vector_db_configs import BaseVectorStoreConfig  # noqa: TC001
from docler.models import ChunkedDocument  # noqa: TC001


class SessionState(State):
    step: int = 1
    document: Document | None = None
    corrected_document: Document | None = None  # Properly added to the state model
    uploaded_file_name: str | None = None
    chunked_doc: ChunkedDocument | None = None
    vector_store_id: str | None = None
    vector_store_name: str | None = None
    vector_provider: str | None = None
    uploaded_chunks: int | None = None
    chunks: list[str] | None = None
    vector_configs: dict[str, BaseVectorStoreConfig] = Field(default_factory=dict)
    converter_configs: dict[str, BaseConverterConfig] = Field(default_factory=dict)
    processor_config: LLMProofReaderConfig | None = None

    def next_step(self):
        """Move to the next step in the workflow."""
        self.step += 1

    def prev_step(self):
        """Move to the previous step in the workflow."""
        self.step -= 1
