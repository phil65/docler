"""Step 1: Document conversion interface."""

from __future__ import annotations

from pathlib import Path
import tempfile

import anyenv
import streambricks as sb
import streamlit as st

from docler.log import get_logger
from docler.streamlit_app.converters import CONVERTERS
from docler.streamlit_app.state import SessionState
from docler.streamlit_app.utils import LANGUAGES, format_image_content


logger = get_logger(__name__)
ALLOWED_EXTENSIONS = ["pdf", "docx", "jpg", "png", "ppt", "pptx", "xls", "xlsx"]


def show_step_1():
    """Show document conversion screen (step 1)."""
    state = SessionState.get()
    st.header("Step 1: Document Conversion")

    # File upload and language selection
    uploaded_file = st.file_uploader("Choose a file", type=ALLOWED_EXTENSIONS)
    selected_converter = st.selectbox(
        "Select converter",
        options=list(CONVERTERS.keys()),
        index=0,
        key="selected_converter",
    )
    language = st.selectbox("Select primary language", options=LANGUAGES, index=0)

    # Get or create converter config
    if selected_converter not in state.converter_configs:
        converter_cls = CONVERTERS[selected_converter]
        state.converter_configs[selected_converter] = converter_cls.Config(
            languages=[language]
        )

    # Edit converter configuration
    with st.expander("Converter Settings", expanded=False):
        config = sb.model_edit(state.converter_configs[selected_converter])
        state.converter_configs[selected_converter] = config

    # Convert document
    if uploaded_file and st.button("Convert Document"):
        with st.spinner(f"Converting with {selected_converter}..."):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                # Create converter from config
                converter = config.get_converter()
                doc = anyenv.run_sync(converter.convert_file(temp_path))
                state.document = doc
                state.uploaded_file_name = uploaded_file.name

                st.success("Document converted successfully!")
                st.button("Proceed to Chunking", on_click=state.next_step)

                # Show document preview
                st.subheader("Document Preview")
                with st.expander("Markdown Content", expanded=False):
                    st.markdown(f"```markdown\n{doc.content}\n```")
                with st.expander("Rendered Content", expanded=True):
                    st.markdown(doc.content)

                # Show images if present
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
                Path(temp_path).unlink()

    # Show preview for already converted document
    elif state.document:
        st.success(f"Document {state.uploaded_file_name!r} already converted!")
        st.button("Proceed to Chunking", on_click=state.next_step)
        st.subheader("Document Preview")
        with st.expander("Markdown Content", expanded=False):
            st.markdown(f"```markdown\n{state.document.content}\n```")
        with st.expander("Rendered Content", expanded=True):
            st.markdown(state.document.content)


if __name__ == "__main__":
    from streambricks import run

    run(show_step_1)
