"""Step 3: Vector Store uploading interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyenv
import streambricks as sb
import streamlit as st

from docler.log import get_logger
from docler.models import ChunkedDocument
from docler.streamlit_app.state import SessionState
from docler.vector_db.dbs import chroma_db, openai_db, pinecone_db


if TYPE_CHECKING:
    from docler.vector_db.base import VectorDB
    from docler.vector_db.base_manager import VectorManagerBase


logger = get_logger(__name__)

VECTOR_STORES: dict[str, type[VectorManagerBase]] = {
    "OpenAI": openai_db.OpenAIVectorManager,
    "Pinecone": pinecone_db.PineconeVectorManager,
    # "Qdrant": qdrant_db.QdrantVectorManager,
    "Chroma": chroma_db.ChromaVectorManager,
}


def show_step_3():
    """Show vector store upload screen (step 3)."""
    state = SessionState.get()
    st.header("Step 3: Upload to Vector Store")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=state.prev_step)

    if not state.chunked_doc:
        st.warning("No chunks to upload. Please go back and chunk a document first.")
        return

    chunked_doc = state.chunked_doc
    chunks = chunked_doc.chunks

    # Vector DB Provider Selection
    st.subheader("Vector Store Configuration")
    provider = st.selectbox(
        "Select Vector Store Provider",
        options=list(VECTOR_STORES.keys()),
        key="selected_vector_provider",
    )

    # Get the manager class for the selected provider
    manager_cls = VECTOR_STORES[provider]

    # Initialize config in the dictionary if not exists
    if provider not in state.vector_configs:
        state.vector_configs[provider] = manager_cls.Config()

    st.divider()
    opts = ["Create new store", "Use existing store"]
    action = st.radio("Vector Store Action", opts, index=0)
    vector_db: VectorDB | None = None

    if action == "Create new store":
        store_name = st.text_input(
            "New Vector Store Name",
            value="docler-store",
            help=f"Name for the new {provider} vector store",
        )

        # Use model_edit to generate the configuration form
        with st.expander("Advanced Configuration", expanded=False):
            state.vector_configs[provider] = sb.model_edit(state.vector_configs[provider])

        if st.button("Create Vector Store"):
            with st.spinner(f"Creating {provider} vector store..."):
                try:
                    manager = manager_cls()
                    config = state.vector_configs[provider]
                    config.collection_name = store_name

                    vector_db = anyenv.run_sync(
                        manager.create_vector_store(
                            store_name, **config.model_dump(exclude={"type"})
                        )
                    )
                    assert vector_db is not None, "Vector store creation failed"
                    state.vector_store_id = vector_db.vector_store_id
                    state.vector_provider = provider
                    st.success(
                        f"{provider} vector store '{store_name}' created successfully!"
                    )
                except Exception as e:
                    st.error(f"Failed to create vector store: {e}")
                    logger.exception("Vector store creation failed")
    else:
        try:
            manager = manager_cls()
            stores = anyenv.run_sync(manager.list_vector_stores())

            if not stores:
                st.info(f"No existing {provider} vector stores found.")
                store_id = st.text_input(
                    f"{provider} Vector Store ID",
                    help=f"ID of an existing {provider} vector store",
                )
            else:
                store_options = {f"{s.name} ({s.db_id})": s.db_id for s in stores}
                store_display = st.selectbox(
                    "Select Vector Store",
                    options=list(store_options.keys()),
                    help=f"Available {provider} vector stores",
                )
                store_id = store_options.get(store_display, "")

            # Use model_edit to generate the configuration form for connection options
            with st.expander("Connection Options", expanded=False):
                state.vector_configs[provider] = sb.model_edit(
                    state.vector_configs[provider]
                )

            if store_id and st.button("Connect to Vector Store"):
                with st.spinner(f"Connecting to {provider} vector store..."):
                    try:
                        manager = manager_cls()
                        config = state.vector_configs[provider]
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(
                                store_id, **config.model_dump(exclude={"type"})
                            )
                        )
                        assert vector_db is not None, "Vector store connection failed"
                        state.vector_store_id = vector_db.vector_store_id
                        state.vector_provider = provider
                        msg = f"Connected to {provider} vector store {store_id!r}!"
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Failed to connect to vector store: {e}")
                        logger.exception("Vector store connection failed")
        except Exception as e:  # noqa: BLE001
            st.error(f"Error listing vector stores: {e}")
            store_id = st.text_input(
                f"{provider} Vector Store ID",
                help=f"ID of an existing {provider} vector store",
            )

            # Use model_edit to generate the configuration form for connection options
            with st.expander("Connection Options", expanded=False):
                state.vector_configs[provider] = sb.model_edit(
                    state.vector_configs[provider]
                )

            if store_id and st.button("Connect to Vector Store"):
                with st.spinner(f"Connecting to {provider} vector store..."):
                    try:
                        manager = manager_cls()
                        config = state.vector_configs[provider]
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(
                                store_id, **config.model_dump(exclude={"type"})
                            )
                        )
                        assert vector_db is not None
                        state.vector_store_id = vector_db.vector_store_id
                        state.vector_provider = provider
                        msg = f"Connected to {provider} vector store {store_id!r}!"
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Failed to connect to vector store: {e}")
                        logger.exception("Vector store connection failed")

    # Upload chunks if we have a connected vector store
    if vec_store_id := state.vector_store_id:
        st.divider()
        st.subheader("Upload Chunks")
        st.write(f"Found {len(chunks)} chunks to upload")

        if st.button("Upload Chunks to Vector Store"):
            with st.spinner("Uploading chunks..."):
                try:
                    provider = state.vector_provider or provider
                    manager_cls = VECTOR_STORES[provider]
                    manager = manager_cls()
                    config = state.vector_configs[provider]
                    vector_db = anyenv.run_sync(
                        manager.get_vector_store(
                            vec_store_id, **config.model_dump(exclude={"type"})
                        )
                    )
                    assert vector_db is not None, "Vector store not found"
                    chunk_ids = anyenv.run_sync(vector_db.add_chunks(chunks))
                    state.uploaded_chunks = len(chunk_ids)
                    msg = f"Uploaded {len(chunk_ids)} chunks to the vector store!"
                    st.success(msg)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    logger.exception("Chunk upload failed")

        # Test vector search if chunks have been uploaded
        if state.uploaded_chunks:
            num = state.uploaded_chunks
            provider = state.vector_provider or provider
            st.success(f"{num} chunks uploaded to {provider} vector db {vec_store_id!r}")

            st.divider()
            st.subheader("Test Your Vector Store")
            query = st.text_input("Enter a query to test your vector store:")
            if query:
                with st.spinner("Searching..."):
                    try:
                        manager_cls = VECTOR_STORES[provider]
                        manager = manager_cls()
                        config = state.vector_configs[provider]
                        cfg = config.model_dump(exclude={"type"})
                        vector_db = anyenv.run_sync(
                            manager.get_vector_store(vec_store_id, **cfg)
                        )
                        assert vector_db is not None, "Vector store not found"
                        results = anyenv.run_sync(vector_db.query(query, k=3))

                        if results:
                            st.write(f"Found {len(results)} relevant chunks:")
                            for i, (chunk, score) in enumerate(results):
                                with st.expander(f"Result {i + 1} - Score: {score:.4f}"):
                                    st.markdown(chunk.text)
                        else:
                            st.info("No results found.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        logger.exception("Vector search failed")


if __name__ == "__main__":
    from streambricks import run

    from docler.models import ChunkedDocument, TextChunk

    state = SessionState.get()
    chunk = TextChunk(text="Sample chunk content", source_doc_id="test", chunk_index=0)
    state.chunked_doc = ChunkedDocument(content="Sample content", chunks=[chunk])
    state.uploaded_file_name = "sample.txt"
    run(show_step_3)
