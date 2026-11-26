"""TextChunk as a pydantic-validated type."""

from pydantic import TypeAdapter

from docler.markdown import Image, TextChunk


def test_textchunk_schema_resolves_on_its_own():
    """Pydantic resolves the dataclass fields from textchunk's own namespace."""
    adapter = TypeAdapter(TextChunk)
    schema = adapter.json_schema()
    assert "Image" in schema["$defs"]
    image = Image(id="img-1", content=b"data", mime_type="image/png")
    chunk = adapter.validate_python({
        "content": "text",
        "source_doc_id": "doc",
        "chunk_index": 0,
        "images": [image],
    })
    assert chunk.images == [image]
