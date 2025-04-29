"""Step 2: Document correction/proof reading interface."""

from __future__ import annotations

import anyenv
import streambricks as sb
import streamlit as st

from docler.configs.processor_configs import LLMProofReaderConfig
from docler.log import get_logger
from docler_streamlit.state import SessionState


logger = get_logger(__name__)


def show_step_2():
    """Show document proof reading/correction screen (step 2)."""
    state = SessionState.get()
    st.header("Step 2: Document Correction")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.button("← Back", on_click=state.prev_step)
    with col3:
        # When proceeding without corrections, copy original document
        def proceed_to_chunking():
            if state.document and not state.corrected_document:
                state.corrected_document = state.document
            state.next_step()

        st.button("Proceed to Chunking →", on_click=proceed_to_chunking)

    if not state.document:
        st.warning("No document to process. Please go back and convert a document first.")
        return

    st.subheader("Proof Reader Configuration")
    if not state.processor_config:
        state.processor_config = LLMProofReaderConfig()
    state.processor_config = sb.model_edit(state.processor_config)
    if st.button("Process Document"):
        with st.spinner("Processing document with AI proof reader..."):
            try:
                processor = state.processor_config.get_provider()
                processed_doc = anyenv.run_sync(processor.process(state.document))
                state.corrected_document = processed_doc
                st.success("Document successfully processed!")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                logger.exception("Document processing failed")

    if state.corrected_document:
        st.subheader("Document Corrections")
        original_content = state.document.content
        corrected_content = state.corrected_document.content
        if original_content == corrected_content:
            st.info("No corrections were made to the document.")
        else:
            from st_diff_viewer import diff_viewer

            diff_viewer(
                original_content,
                corrected_content,
                split_view=True,
                title="OCR Corrections",
            )
            if "proof_reading" in state.corrected_document.metadata:
                proof_metadata = state.corrected_document.metadata["proof_reading"]
                col1, col2 = st.columns(2)
                with col1:
                    count = proof_metadata.get("corrections_count", 0)
                    st.metric("Total Corrections", count)
                with col2:
                    if count:
                        lines = original_content.splitlines()
                        percentage = min(100, (count / len(lines)) * 100)
                        st.metric("% Lines Corrected", f"{percentage:.1f}%")
                if proof_metadata.get("corrections"):
                    st.subheader("Line Corrections")
                    for correction in proof_metadata["corrections"]:
                        with st.expander(f"Line {correction.get('line_number')}"):
                            st.code(correction.get("corrected"), language="markdown")

        st.subheader("Processed Document Preview")
        tabs = st.tabs(["Rendered Content", "Raw Markdown"])
        with tabs[0]:
            st.markdown(state.corrected_document.content)
        with tabs[1]:
            st.code(state.corrected_document.content, language="markdown")
    else:
        st.info(
            "You can proceed to chunking without corrections, "
            "or use the processor to correct common OCR errors."
        )
        with st.expander("View Original Document"):
            st.markdown(state.document.content)


if __name__ == "__main__":
    from streambricks import run

    # For testing purposes
    run(show_step_2)
