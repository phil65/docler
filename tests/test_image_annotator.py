"""AIImageAnnotator concurrency with a fake per-image call."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from docler.annotators.ai_image_annotator import AIImageAnnotator
from docler.markdown import Image, TextChunk
from docler.models import ChunkedDocument


if TYPE_CHECKING:
    import pytest


async def test_images_of_all_chunks_are_annotated_concurrently(monkeypatch: pytest.MonkeyPatch):
    running = 0
    max_running = 0
    seen: list[str] = []

    async def fake_process(self: AIImageAnnotator, image: Image) -> Image:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0.01)
        running -= 1
        seen.append(image.id)
        return image

    monkeypatch.setattr(AIImageAnnotator, "_process_image", fake_process)
    images = [Image(id=f"img{i}", content=b"x", mime_type="image/png") for i in range(4)]
    chunks = [
        TextChunk(content="t", source_doc_id="d", chunk_index=i, images=[img])
        for i, img in enumerate(images)
    ]
    chunks.append(TextChunk(content="t", source_doc_id="d", chunk_index=4, images=[images[0]]))
    doc = ChunkedDocument(content="t", chunks=chunks)
    await AIImageAnnotator(batch_size=3).annotate(doc)
    assert sorted(seen) == ["img0", "img1", "img2", "img3"]
    assert max_running == 3  # noqa: PLR2004


async def test_document_annotator_keeps_results_when_one_chunk_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    import sys
    import types

    from docler.annotators.ai_document_annotator import AIDocumentAnnotator

    class FakeAgent:
        def __init__(self, **_: object) -> None:
            pass

        async def run(self, prompt: str) -> object:
            if "BAD" in prompt:
                raise RuntimeError("llm failed")
            dump = types.SimpleNamespace(model_dump=lambda: {"annotated": True})
            return types.SimpleNamespace(content=dump)

    monkeypatch.setitem(sys.modules, "agentpool", types.SimpleNamespace(Agent=FakeAgent))
    chunks = [
        TextChunk(content=text, source_doc_id="d", chunk_index=i)
        for i, text in enumerate(["good", "BAD", "fine"])
    ]
    doc = ChunkedDocument(content="t", chunks=chunks)
    await AIDocumentAnnotator(batch_size=5).annotate(doc)
    assert [c.metadata.get("annotated") for c in doc.chunks] == [True, None, True]
