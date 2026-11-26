"""Tests for directory conversion and fan-out with a fake converter."""

import asyncio
from typing import TYPE_CHECKING, ClassVar

from docler.converters.base import ConverterResult, DocumentConverter
from docler.converters.dir_converter import DirectoryConverter


if TYPE_CHECKING:
    import io
    from pathlib import Path

    from schemez import MimeType


class TextConverter(DocumentConverter):
    """Converts text files, fails on content "fail", tracks concurrency."""

    NAME = "test_text"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"text/plain"}

    def __init__(self) -> None:
        super().__init__()
        self.running = 0
        self.max_running = 0

    async def _convert_async(self, data: io.BytesIO, mime_type: MimeType) -> ConverterResult:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(0.01)
            text = data.getvalue().decode()
            if text == "fail":
                raise RuntimeError("conversion failed")
            return ConverterResult(content=text)
        finally:
            self.running -= 1


def _make_files(tmp_path: Path, contents: list[str]) -> None:
    for i, content in enumerate(contents):
        (tmp_path / f"f{i}.txt").write_text(content)


async def test_one_failure_keeps_the_other_results(tmp_path: Path):
    _make_files(tmp_path, ["a", "fail", "c"])
    states = [
        state async for state in DirectoryConverter(TextConverter()).convert_with_progress(tmp_path)
    ]
    final = states[-1]
    assert (final.successful_files, final.failed_files) == (2, 1)
    assert set(final.errors) == {"f1.txt"}
    assert {k: d.content for k, d in final.results.items()} == {"f0.txt": "a", "f2.txt": "c"}


async def test_convert_directory_returns_successes(tmp_path: Path):
    _make_files(tmp_path, ["a", "fail", "c"])
    results = await TextConverter().convert_directory(tmp_path)
    assert {k: d.content for k, d in results.items()} == {"f0.txt": "a", "f2.txt": "c"}


async def test_directory_concurrency_is_bounded(tmp_path: Path):
    _make_files(tmp_path, [str(i) for i in range(12)])
    converter = TextConverter()
    states = [
        s async for s in DirectoryConverter(converter, chunk_size=3).convert_with_progress(tmp_path)
    ]
    assert states[-1].successful_files == 12  # noqa: PLR2004
    assert converter.max_running == 3  # noqa: PLR2004


async def test_convert_files_concurrency_is_bounded(tmp_path: Path):
    _make_files(tmp_path, [str(i) for i in range(30)])
    converter = TextConverter()
    docs = await converter.convert_files(sorted(tmp_path.iterdir()))
    assert len(docs) == 30  # noqa: PLR2004
    assert converter.max_running <= converter.MAX_CONCURRENCY
