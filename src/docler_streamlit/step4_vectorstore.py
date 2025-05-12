"""Step 4: Vector Store uploading interface."""

from __future__ import annotations

from typing import Any

import anyenv
from mkdown import TextChunk
import numpy as np
import streamlit as st

from docler.embeddings.openai_provider import OpenAIEmbeddings
from docler.log import get_logger
from docler.vector_db.dbs.pinecone_db import PineconeVectorManager
from docler_streamlit.state import SessionState


logger = get_logger(__name__)


class VectorStore:
    """Helper class to manage both embedding and vector storage."""

    def __init__(self, store_name: str, backend):
        """Initialize with a vector store backend."""
        self.store_name = store_name
        self.backend = backend
        # Use OpenAI embeddings instead of LiteLLM
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    async def add_chunks(self, chunks: list[TextChunk]) -> list[str]:
        """Embed and add chunks to the vector store."""
        vectors = []
        metadata_list = []

        for chunk in chunks:
            embedding = await self.embedder.embed_query(chunk.content)
            vectors.append(embedding)
            metadata = {
                "text": chunk.content,
                "source_doc_id": chunk.source_doc_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            metadata_list.append(metadata)

        # Add vectors to storage
        return await self.backend.add_vectors(
            vectors=np.array(vectors),
            metadata=metadata_list,
        )

    async def query(
        self, query: str, k: int = 4, filters: dict[str, Any] | None = None
    ) -> list[tuple[TextChunk, float]]:
        """Search for similar chunks."""
        # Embed the query
        query_embedding = await self.embedder.embed_query(query)

        # Search the vector store
        results = await self.backend.search_vectors(
            query_vector=query_embedding,
            k=k,
            filters=filters,
        )

        # Convert search results to TextChunks
        chunks_with_scores = []
        for result in results:
            metadata = result.metadata
            content = metadata.pop("text", "") or result.text or ""
            source_doc_id = metadata.pop("source_doc_id", "")
            chunk_index = metadata.pop("chunk_index", 0)

            # Create a TextChunk from the search result
            chunk = TextChunk(
                content=content,
                source_doc_id=source_doc_id,
                chunk_index=chunk_index,
                metadata=metadata,
            )
            chunks_with_scores.append((chunk, result.score))

        return chunks_with_scores


def show_step_4():
    """Show vector store upload screen (step 4)."""
    state = SessionState.get()
    st.header("Step 4: Upload to Vector Store")

    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Back", on_click=state.prev_step)

    if not state.chunked_doc:
        st.warning("No chunks to upload. Please go back and chunk a document first.")
        return

    chunked_doc = state.chunked_doc
    chunks = chunked_doc.chunks

    try:
        # Initialize Pinecone manager (uses PINECONE_API_KEY environment variable)
        manager = PineconeVectorManager()

        # Load existing vector stores
        with st.spinner("Loading vector stores..."):
            stores = anyenv.run_sync(manager.list_vector_stores())

        # Display vector store options
        col1, col2 = st.columns(2)
        with col1:
            store_options = ["Create new store"] + [s.name for s in stores]
            selected_option = st.selectbox("Vector Store", options=store_options)

        with col2:
            if selected_option == "Create new store":
                store_name = st.text_input("New Store Name", value="docler-store")
                if st.button("Create Store"):
                    with st.spinner(f"Creating store '{store_name}'..."):
                        try:
                            anyenv.run_sync(manager.create_vector_store(store_name))
                            # Store the name only
                            state.vector_store_name = store_name
                            st.success(f"Store '{store_name}' created!")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to create store: {e}")
            else:
                # Store the selected store name
                state.vector_store_name = selected_option
                st.info(f"Using existing store: {selected_option}")

        # Upload button (only show if we have a store name)
        if state.vector_store_name:
            st.divider()
            st.write(f"Found {len(chunks)} chunks to upload")

            if st.button("Upload Chunks"):
                with st.spinner("Uploading chunks..."):
                    try:
                        # Get the backend for the selected store
                        backend = anyenv.run_sync(
                            manager.get_vector_store(state.vector_store_name)
                        )

                        # Create our helper class
                        vector_store = VectorStore(state.vector_store_name, backend)

                        # Upload the chunks
                        chunk_ids = anyenv.run_sync(vector_store.add_chunks(chunks))
                        state.uploaded_chunks = len(chunk_ids)
                        st.success(f"Uploaded {len(chunk_ids)} chunks!")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        logger.exception("Chunk upload failed")

            # Test search (only show if chunks were uploaded)
            if state.uploaded_chunks:
                st.divider()
                st.subheader("Test Search")
                query = st.text_input("Enter a query:")

                if query:
                    with st.spinner("Searching..."):
                        try:
                            # Get the backend for the selected store
                            backend = anyenv.run_sync(
                                manager.get_vector_store(state.vector_store_name)
                            )

                            # Create our helper class
                            vector_store = VectorStore(state.vector_store_name, backend)

                            # Search for results
                            results = anyenv.run_sync(vector_store.query(query, k=3))

                            if results:
                                for i, (chunk, score) in enumerate(results):
                                    with st.expander(
                                        f"Result {i + 1} - Score: {score:.4f}"
                                    ):
                                        st.markdown(chunk.content)
                            else:
                                st.info("No results found.")
                        except Exception as e:
                            st.error(f"Search failed: {e}")
                            logger.exception("Vector search failed")

    except Exception as e:  # noqa: BLE001
        st.error(f"Error connecting to Pinecone: {e}")
        st.info("Make sure the PINECONE_API_KEY environment variable is set.")
