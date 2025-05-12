"""Streamlit app for markdown chunking visualization."""

from __future__ import annotations

import anyenv
from mkdown import Document
import streamlit as st

from docler.chunkers.markdown_chunker import MarkdownChunker
from docler_streamlit.utils import display_chunk_preview


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


def main():
    """Main Streamlit app."""
    st.title("Markdown Chunker")
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

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Input Markdown")
    with col2:
        if st.button("Load Sample"):
            st.session_state.markdown_input = SAMPLE_MARKDOWN
    input_text = st.session_state.get("markdown_input", "")
    text = "Enter your markdown here:"
    markdown_input = st.text_area(text, value=input_text, height=200)

    if markdown_input:
        doc = Document(content=markdown_input, source_path="input.md")
        chunker = MarkdownChunker(
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            chunk_overlap=overlap,
        )
        if st.button("Chunk Markdown"):
            with st.spinner("Processing..."):
                chunked = anyenv.run_sync(chunker.chunk(doc))
                st.subheader("Chunks")
                for chunk in chunked.chunks:
                    display_chunk_preview(chunk, expanded=True)


if __name__ == "__main__":
    from streambricks import run

    run(main)
