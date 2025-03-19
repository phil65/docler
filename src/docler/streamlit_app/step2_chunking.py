"""Step 2: Document chunking interface."""

from __future__ import annotations

import logging
from typing import Literal, cast

import anyenv
import streamlit as st

from docler.chunkers.ai_chunker import AIChunker
from docler.chunkers.base import TextChunk, TextChunker
from docler.chunkers.llamaindex_chunker import LlamaIndexChunker
from docler.chunkers.markdown_chunker import MarkdownChunker
from docler.models import Document
from docler.streamlit_app.chunkers import CHUNKERS
from docler.streamlit_app.state import prev_step
from docler.streamlit_app.utils import format_image_content


logger = logging.getLogger(__name__)


def show_step_2():
    """Show document chunking screen (step 2)."""
    st.header("Step 2: Document Chunking")

    # Navigation buttons at the top
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=prev_step)

    # Check if we have a document to chunk
    if not st.session_state.document:
        st.warning("No document to chunk. Please go back and convert a document first.")
        return

    doc = cast(Document, st.session_state.document)

    # Chunker selection and configuration
    st.subheader("Chunking Configuration")

    chunker_type = st.selectbox(
        "Select chunker",
        options=list(CHUNKERS.keys()),
        key="selected_chunker",
    )

    # Display chunker-specific settings
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
            chunker_subtype = st.selectbox(
                "LlamaIndex chunker type",
                options=["markdown", "sentence", "token", "fixed"],
                index=0,
            )
        with col2:
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
            )

        chunker = LlamaIndexChunker(
            chunker_type=cast(
                Literal["sentence", "token", "fixed", "markdown"], chunker_subtype
            ),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),  # 10% overlap
        )

    elif chunker_type == "AI":
        model = st.selectbox(
            "LLM Model",
            options=[
                "openrouter:openai/o3-mini",
                "openrouter:google/gemini-2.0-flash-lite-001",
            ],
            index=0,
        )

        chunker = AIChunker(model=model)

    # Process button
    if chunker and st.button("Chunk Document"):
        with st.spinner("Processing document..."):
            try:
                chunks = anyenv.run_sync(chunker.split(doc))
                st.session_state.chunks = chunks
                st.success(f"Document successfully chunked into {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Chunking failed: {e}")
                logger.exception("Chunking failed")

    # Display chunks if available
    if st.session_state.chunks:
        chunks = cast(list[TextChunk], st.session_state.chunks)
        st.subheader(f"Chunks ({len(chunks)})")

        # Add filtering option
        filter_text = st.text_input("Filter chunks by content:", "")

        # Display chunks
        for i, chunk in enumerate(chunks):
            # Skip if doesn't match filter
            if filter_text and filter_text.lower() not in chunk.text.lower():
                continue

            # Create expander for chunk
            header_text = f"Chunk {i + 1}"
            if chunk.metadata.get("header"):
                header_text += f" - {chunk.metadata['header']}"
            header_text += f" ({len(chunk.text)} chars)"

            with st.expander(header_text, expanded=i == 0):
                # Create tabs for different views
                raw_tab, rendered_tab, debug_tab, images_tab = st.tabs([
                    "Raw",
                    "Rendered",
                    "Debug Info",
                    "Images",
                ])

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
