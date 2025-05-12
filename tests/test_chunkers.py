"""Tests for the markdown chunker."""

from __future__ import annotations

from mkdown import Document, Image
import pytest

from docler.chunkers.markdown_chunker import MarkdownChunker


MIME = "image/png"


@pytest.fixture
def sample_images() -> list[Image]:
    """Create sample images for testing."""
    return [
        Image(id="img1", content=b"fake-image-1", mime_type=MIME, filename="image1.png"),
        Image(id="img2", content=b"fake-image-2", mime_type=MIME, filename="image2.png"),
        Image(id="img3", content=b"fake-image-3", mime_type=MIME, filename="image3.png"),
    ]


@pytest.fixture
def markdown_with_images(sample_images: list[Image]) -> Document:
    """Create a markdown document with image references."""
    content = """# Section 1

This is some text with an image:
![img1](image1.png)
More text here.

## Subsection

Another image here:
![img2](image2.png)

# Section 2

Final section with the last image:
![img3](image3.png)
"""
    return Document(
        content=content,
        images=sample_images,
        source_path="test.md",
    )


@pytest.fixture
def long_markdown_with_images(sample_images: list[Image]) -> Document:
    """Create a long markdown document to test size-based splitting."""
    # Create content that exceeds max_chunk_size but contains images
    content = (
        "# Long Section\n\n"
        + "Text " * 200  # Some text before image
        + "\n![img1](image1.png)\n"
        + "Text " * 200  # Text between images
        + "\n![img2](image2.png)\n"
        + "Text " * 200  # More text
        + "\n![img3](image3.png)\n"
        + "Text " * 200  # Final text
    )
    return Document(content=content, images=sample_images, source_path="test.md")


async def test_header_based_image_splitting(markdown_with_images: Document):
    """Test that images stay with their relevant sections in header-based splitting."""
    chunker = MarkdownChunker()
    chunks = await chunker.split(markdown_with_images)

    # We expect 3 chunks based on the headers
    assert len(chunks) == 3  # noqa: PLR2004

    # First chunk should contain image1
    assert len(chunks[0].images) == 1
    assert chunks[0].images[0].id == "img1"
    assert "![img1](image1.png)" in chunks[0].content

    # Second chunk (subsection) should contain image2
    assert len(chunks[1].images) == 1
    assert chunks[1].images[0].id == "img2"
    assert "![img2](image2.png)" in chunks[1].content

    # Third chunk should contain image3
    assert len(chunks[2].images) == 1
    assert chunks[2].images[0].id == "img3"
    assert "![img3](image3.png)" in chunks[2].content


async def test_size_based_image_splitting(long_markdown_with_images: Document):
    """Test that images are handled correctly in size-based splitting."""
    # Use smaller chunk size to force size-based splitting
    chunker = MarkdownChunker(max_chunk_size=500)
    chunks = await chunker.split(long_markdown_with_images)

    # Verify that images appear in appropriate chunks
    for chunk in chunks:
        # Each chunk should only contain the images referenced within its text
        for image in chunk.images:
            ref = f"![{image.id}]({image.filename})"
            assert ref in chunk.content, f"Image reference {ref} not found in chunk"


async def test_headerless_content_with_images():
    """Test handling of content without headers but with images."""
    content = (
        "This is a document without headers.\n"
        "It contains an image here:\n"
        "![img1](image1.png)\n"
        "And some more text...\n"
        "And another image:\n"
        "![img2](image2.png)"
    )

    images = [
        Image(id="img1", content=b"test1", mime_type=MIME, filename="image1.png"),
        Image(id="img2", content=b"test2", mime_type=MIME, filename="image2.png"),
    ]

    doc = Document(content=content, images=images, source_path="test.md")
    chunker = MarkdownChunker(max_chunk_size=100)  # Small size to force splitting
    chunks = await chunker.split(doc)

    # Verify that each image appears in the correct chunk
    for chunk in chunks:
        for image in chunk.images:
            ref = f"![{image.id}]({image.filename})"
            assert ref in chunk.content, f"Image reference {ref} not found in chunk"


async def test_image_metadata_preservation():
    """Test that image metadata is preserved correctly in chunks."""
    content = """\
# Section with image
![test](test.png)
Some text here.
"""
    image = Image(id="test", content=b"test-content", mime_type=MIME, filename="test.png")

    doc = Document(content=content, images=[image], source_path="test.md")
    chunker = MarkdownChunker()
    chunks = await chunker.split(doc)

    # Verify image properties are preserved
    assert len(chunks) == 1
    assert len(chunks[0].images) == 1
    chunk_image = chunks[0].images[0]
    assert chunk_image.id == image.id
    assert chunk_image.content == image.content
    assert chunk_image.mime_type == image.mime_type
    assert chunk_image.filename == image.filename


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
