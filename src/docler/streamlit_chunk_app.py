"""Streamlit app for markdown chunking visualization."""

from __future__ import annotations

import anyenv
import streamlit as st

from docler.chunkers.markdown_chunker import MarkdownChunker
from docler.models import Document


SAMPLE_MARKDOWN = """# Introduction

This is a sample markdown document to test chunking.

## First Section

Some content here with an example image:
![example](image.png)

### Subsection

More detailed content...

## Second Section

Another block of text that demonstrates how chunking works.
"""


def main() -> None:
    """Main Streamlit app."""
    st.title("Markdown Chunker")

    # Sidebar for settings
    with st.sidebar:
        st.header("Chunker Settings")
        min_size = st.number_input(
            "Minimum chunk size",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Minimum characters per chunk",
        )
        max_size = st.number_input(
            "Maximum chunk size",
            min_value=100,
            max_value=2000,
            value=1500,
            step=100,
            help="Maximum characters per chunk",
        )
        overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Character overlap between chunks",
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Input Markdown")

    # Add sample button in second column
    with col2:
        if st.button("Load Sample"):
            st.session_state.markdown_input = SAMPLE_MARKDOWN

    # Text area for markdown input
    input_text = st.session_state.get("markdown_input", "")
    text = "Enter your markdown here:"
    markdown_input = st.text_area(text, value=input_text, height=200)

    if markdown_input:
        # Create document and chunk it
        doc = Document(content=markdown_input, source_path="input.md")
        chunker = MarkdownChunker(
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            chunk_overlap=overlap,
        )

        # Process chunks
        if st.button("Chunk Markdown"):
            with st.spinner("Processing..."):
                chunks = anyenv.run_sync(chunker.split(doc))

                # Display chunks
                st.subheader("Chunks")

                for i, chunk in enumerate(chunks):
                    # Create expander for chunk
                    header_text = f"Chunk {i + 1}"
                    if chunk.metadata.get("header"):
                        header_text += f" - {chunk.metadata['header']}"
                    header_text += f" ({len(chunk.text)} chars)"

                    with st.expander(header_text, expanded=True):
                        # Create tabs for different views
                        raw_tab, rendered_tab, debug_tab = st.tabs([
                            "Raw",
                            "Rendered",
                            "Debug Info",
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


if __name__ == "__main__":
    main()
