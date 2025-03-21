"""Chunkers configuration for the Streamlit app."""

from __future__ import annotations

from docler.chunkers.ai_chunker import AIChunker
from docler.chunkers.llamaindex_chunker import LlamaIndexChunker
from docler.chunkers.markdown_chunker import MarkdownChunker


CHUNKERS = {"Markdown": MarkdownChunker, "LlamaIndex": LlamaIndexChunker, "AI": AIChunker}
