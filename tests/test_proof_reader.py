"""Tests for LLMProofReader with a fake LLM."""

from __future__ import annotations

import re
import sys
import types
from typing import Any

import pytest

from docler.markdown import Document, TextChunk
from docler.processors.ai_processor import LineCorrection, LLMProofReader, chunk_line_spans
from docler_config.chunker_configs import MarkdownChunkerConfig


class FakeTalk:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def extract_multiple(self, text: str, **_: Any) -> list[LineCorrection]:
        """Correct every numbered line to its upper-case version."""
        self.calls.append(text)
        return [
            LineCorrection(line_number=int(m[1]), corrected=m[2].upper())
            for m in re.finditer(r"^\s*(\d+) \| (.*)$", text, re.MULTILINE)
        ]


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    class FakeAgent:
        def __init__(self, **_: Any) -> None:
            self.talk = FakeTalk(calls)

        def __class_getitem__(cls, _: Any) -> type[FakeAgent]:
            return cls

    monkeypatch.setitem(sys.modules, "agentpool", types.SimpleNamespace(Agent=FakeAgent))
    return calls


async def test_chunk_corrections_hit_their_own_lines(llm_calls: list[str]):
    content = "\n".join(f"line {i:02d}" for i in range(1, 13))  # 7 chars + newline each
    config = MarkdownChunkerConfig(min_chunk_size=1, max_chunk_size=32, chunk_overlap=0)
    reader = LLMProofReader(chunker=config, include_diffs=False)
    result = await reader.process(Document(content=content, source_path="x.md"))
    assert len(llm_calls) > 1
    assert result.content == content.upper()
    assert result.metadata["proof_reading"]["corrected_lines"] == list(range(12))


async def test_markdown_sections_hit_their_own_lines(llm_calls: list[str]):
    content = "# A\nfoo bar\nmore\n\n# B\nbaz qux"
    reader = LLMProofReader(chunker=MarkdownChunkerConfig(min_chunk_size=1), include_diffs=False)
    result = await reader.process(Document(content=content, source_path="x.md"))
    assert len(llm_calls) == 2  # noqa: PLR2004
    assert result.content == "# A\nFOO BAR\nMORE\n\n# B\nBAZ QUX"


def test_chunk_that_is_not_in_the_document_raises():
    chunk = TextChunk(content="rewritten", source_doc_id="x", chunk_index=0)
    with pytest.raises(ValueError, match="line positions"):
        chunk_line_spans("original\ntext", [chunk])
