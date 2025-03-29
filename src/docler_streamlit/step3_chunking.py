"""Step 3: Document chunking interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import anyenv
import streambricks as sb
import streamlit as st

from docler.chunkers.ai_chunker import AIChunker
from docler.chunkers.llamaindex_chunker import LlamaIndexChunker
from docler.chunkers.markdown_chunker import MarkdownChunker
from docler.configs import DEFAULT_CHUNKER_SYSTEM_PROMPT
from docler.log import get_logger
from docler_streamlit.chunkers import CHUNKERS
from docler_streamlit.state import SessionState
from docler_streamlit.utils import display_chunk_preview


if TYPE_CHECKING:
    from docler.chunkers.base import TextChunker


logger = get_logger(__name__)


def show_step_3():
    """Show document chunking screen (step 3)."""
    st.header("Step 3: Document Chunking")
    state = SessionState.get()
    doc = state.corrected_document if state.corrected_document else state.document
    # Navigation buttons at the top
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=state.prev_step)
    if not doc:
        st.warning("No document to chunk. Please go back and convert a document first.")
        return
    st.subheader("Chunking Configuration")
    opts = list(CHUNKERS.keys())
    chunker_type = st.selectbox("Select chunker", options=opts, key="selected_chunker")
    chunker: TextChunker | None = None
    if chunker_type == "Markdown":
        col1, col2, col3 = st.columns(3)
        with col1:
            min_size = st.number_input(
                "Minimum chunk size",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
            )
        with col2:
            max_size = st.number_input(
                "Maximum chunk size",
                min_value=100,
                max_value=5000,
                value=1500,
                step=100,
            )
        with col3:
            overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
            )

        chunker = MarkdownChunker(
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            chunk_overlap=overlap,
        )

    elif chunker_type == "LlamaIndex":
        col1, col2 = st.columns(2)
        with col1:
            opts = ["markdown", "sentence", "token", "fixed"]
            chunker_subtype = st.selectbox("Chunker type", options=opts, index=0)
        with col2:
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
            )
        typ = cast(Literal["sentence", "token", "fixed", "markdown"], chunker_subtype)
        chunker = LlamaIndexChunker(
            chunker_type=typ,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),  # 10% overlap
        )

    elif chunker_type == "AI":
        model = sb.model_selector(providers=["openrouter"])
        model_name = model.pydantic_ai_id if model else None
        sys_prompt = st.text_area("System prompt", value=DEFAULT_CHUNKER_SYSTEM_PROMPT)
        chunker = AIChunker(model=model_name, system_prompt=sys_prompt)
    if chunker and st.button("Chunk Document"):
        with st.spinner("Processing document..."):
            try:
                chunked = anyenv.run_sync(chunker.chunk(doc))
                state.chunked_doc = chunked
                st.success(
                    f"Document successfully chunked into {len(chunked.chunks)} chunks!"
                )
            except Exception as e:
                st.error(f"Chunking failed: {e}")
                logger.exception("Chunking failed")

    if state.chunked_doc:
        st.button("Proceed to Vector Store Upload", on_click=state.next_step)
        chunked = state.chunked_doc
        chunks = chunked.chunks
        st.subheader(f"Chunks ({len(chunks)})")
        filter_text = st.text_input("Filter chunks by content:", "")
        for i, chunk in enumerate(chunks):
            if filter_text and filter_text.lower() not in chunk.content.lower():
                continue
            display_chunk_preview(chunk, expanded=(i == 0))
