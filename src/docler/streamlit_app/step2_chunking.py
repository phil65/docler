"""Step 2: Document chunking interface."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import anyenv
import streambricks as sb
import streamlit as st

from docler.chunkers.ai_chunker import AIChunker
from docler.chunkers.ai_chunker.chunker import SYS_PROMPT
from docler.chunkers.llamaindex_chunker import LlamaIndexChunker
from docler.chunkers.markdown_chunker import MarkdownChunker
from docler.models import Document, TextChunk
from docler.streamlit_app.chunkers import CHUNKERS
from docler.streamlit_app.state import next_step, prev_step
from docler.streamlit_app.utils import format_image_content


if TYPE_CHECKING:
    from docler.chunkers.base import TextChunker


logger = logging.getLogger(__name__)


def show_step_2():
    """Show document chunking screen (step 2)."""
    st.header("Step 2: Document Chunking")

    # Navigation buttons at the top
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=prev_step)
    if not st.session_state.document:
        st.warning("No document to chunk. Please go back and convert a document first.")
        return
    doc = cast(Document, st.session_state.document)
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
        sys_prompt = st.text_area("System prompt", value=SYS_PROMPT)
        chunker = AIChunker(model=model_name, system_prompt=sys_prompt)
    if chunker and st.button("Chunk Document"):
        with st.spinner("Processing document..."):
            try:
                chunks = anyenv.run_sync(chunker.split(doc))
                st.session_state.chunks = chunks
                st.success(f"Document successfully chunked into {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Chunking failed: {e}")
                logger.exception("Chunking failed")

    if st.session_state.chunks:
        st.button("Proceed to Vector Store Upload", on_click=next_step)
        chunks = cast(list[TextChunk], st.session_state.chunks)
        st.subheader(f"Chunks ({len(chunks)})")
        filter_text = st.text_input("Filter chunks by content:", "")
        for i, chunk in enumerate(chunks):
            if filter_text and filter_text.lower() not in chunk.text.lower():
                continue

            header_text = f"Chunk {i + 1}"
            if chunk.metadata.get("header"):
                header_text += f" - {chunk.metadata['header']}"
            header_text += f" ({len(chunk.text)} chars)"
            with st.expander(header_text, expanded=i == 0):
                tabs = ["Raw", "Rendered", "Debug Info", "Images"]
                raw_tab, rendered_tab, debug_tab, images_tab = st.tabs(tabs)
                with raw_tab:
                    st.code(chunk.text, language="markdown")
                with rendered_tab:
                    st.markdown(chunk.text)
                with debug_tab:
                    debug_info = {
                        "Chunk Index": chunk.chunk_index,
                        "Source": chunk.source_doc_id,
                        "Images": len(chunk.images),
                        **chunk.metadata,
                    }
                    st.json(debug_info)

                with images_tab:
                    if not chunk.images:
                        st.info("No images in this chunk")
                    else:
                        for image in chunk.images:
                            data_url = format_image_content(
                                image.content, image.mime_type
                            )
                            st.markdown(f"**ID:** {image.id}")
                            if image.filename:
                                st.markdown(f"**Filename:** {image.filename}")
                            st.markdown(f"**MIME Type:** {image.mime_type}")
                            st.image(data_url)
                            st.divider()
