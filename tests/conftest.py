"""Test configuration for Docler."""

from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).parent
TEST_RESOURCES = TESTS_DIR / "resources"


@pytest.fixture
def resources_dir() -> Path:
    return TEST_RESOURCES


@pytest.fixture
def sample_markdown_doc():
    """Return a sample markdown document for testing."""
    from docler.models import Document

    content = """# Introduction

This is an introduction to the document.
It covers several topics.

## First Section

The first section goes into detail about the topic.
More information is provided here.

## Second Section

The second section provides examples and use cases.
Examples help understand the concepts better.

### Subsection

This is a subsection with more specific details.
"""
    return Document(content=content, source_path="test_doc.md")
