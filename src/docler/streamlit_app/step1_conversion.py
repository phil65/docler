"""Step 1: Document conversion interface."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

import anyenv
import streamlit as st

from docler.streamlit_app.converters import CONVERTERS
from docler.streamlit_app.state import next_step
from docler.streamlit_app.utils import LANGUAGES, format_image_content


logger = logging.getLogger(__name__)


def show_step_1():
    """Show document conversion screen (step 1)."""
    st.header("Step 1: Document Conversion")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "jpg", "png", "ppt", "pptx", "xls", "xlsx"]
    )

    # Converter selection
    selected_converter = st.selectbox(
        "Select converter",
        options=list(CONVERTERS.keys()),
        index=0,
        key="selected_converter",
    )

    # Language selection
    language = st.selectbox(
        "Select primary language",
        options=LANGUAGES,
        index=0,
    )

    # Only show conversion button if a file is uploaded
    if uploaded_file and st.button("Convert Document"):
        with st.spinner(f"Converting with {selected_converter}..."):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix, delete=False
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                # Convert document
                converter_cls = CONVERTERS[selected_converter]
                converter = converter_cls(languages=[language])
                doc = anyenv.run_sync(converter.convert_file(temp_path))

                # Store in session state
                st.session_state.document = doc
                st.session_state.uploaded_file_name = uploaded_file.name

                # Show success and navigation button
                st.success("Document converted successfully!")
                st.button("Proceed to Chunking", on_click=next_step)

                # Show document preview
                st.subheader("Document Preview")
                with st.expander("Markdown Content", expanded=False):
                    st.markdown(f"```markdown\n{doc.content}\n```")

                with st.expander("Rendered Content", expanded=True):
                    st.markdown(doc.content)

                # Show image preview if available
                if doc.images:
                    with st.expander(f"Images ({len(doc.images)})", expanded=False):
                        for image in doc.images:
                            data_url = format_image_content(
                                image.content, image.mime_type
                            )
                            st.markdown(f"**ID:** {image.id}")
                            if image.filename:
                                st.markdown(f"**Filename:** {image.filename}")
                            st.markdown(f"**MIME Type:** {image.mime_type}")
                            st.image(data_url)
                            st.divider()

            except Exception as e:
                st.error(f"Conversion failed: {e!s}")
                logger.exception("Conversion failed")
            finally:
                # Clean up temp file
                Path(temp_path).unlink()

    # If document already converted, show preview and navigation
    elif st.session_state.document:
        st.success(f"Document '{st.session_state.uploaded_file_name}' already converted!")
        st.button("Proceed to Chunking", on_click=next_step)

        # Show document preview
        st.subheader("Document Preview")
        with st.expander("Markdown Content", expanded=False):
            st.markdown(f"```markdown\n{st.session_state.document.content}\n```")

        with st.expander("Rendered Content", expanded=True):
            st.markdown(st.session_state.document.content)
