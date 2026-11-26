"""Directory conversion helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import mimetypes
from typing import TYPE_CHECKING

from upathtools import list_files, to_upath

from docler.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import upath
    from upath.types import JoinablePathLike

    from docler.converters.base import DocumentConverter
    from docler.markdown import Document


logger = get_logger(__name__)


@dataclass
class Conversion:
    """Represents the state of a directory conversion."""

    total_files: int
    """Total number of files to convert."""

    processed_files: int
    """Number of files processed so far."""

    current_file: str
    """Path of the file currently being processed."""

    successful_files: int
    """Number of files successfully converted."""

    failed_files: int
    """Number of files that failed to convert."""

    errors: dict[str, Exception] = field(default_factory=dict)
    """Map of file paths to conversion errors."""

    results: dict[str, Document] = field(default_factory=dict)
    """Map of file paths to converted documents."""


class DirectoryConverter:
    """Directory conversion functionality."""

    def __init__(self, converter: DocumentConverter, *, chunk_size: int = 50) -> None:
        """Initialize the directory converter.

        Args:
            converter: Document converter to use.
            chunk_size: Number of files to convert in parallel.
        """
        self.converter = converter
        self.chunk_size = chunk_size

    async def convert(
        self,
        directory: JoinablePathLike,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
        pdf_password: str | None = None,
    ) -> dict[str, Document]:
        """Convert all supported files in a directory.

        Files that fail to convert are left out (and logged).

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.
            pdf_password: Password for encrypted PDF files.

        Returns:
            Mapping of relative paths to converted documents.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        state: Conversion | None = None
        async for current in self.convert_with_progress(
            directory,
            pattern=pattern,
            recursive=recursive,
            exclude=exclude,
            max_depth=max_depth,
            pdf_password=pdf_password,
        ):
            state = current
        if state is None:
            return {}
        for path, error in state.errors.items():
            logger.warning("Failed to convert %s: %s", path, error)
        return state.results

    async def convert_with_progress(
        self,
        directory: JoinablePathLike,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
        pdf_password: str | None = None,
    ) -> AsyncIterator[Conversion]:
        """Convert files with progress updates.

        Converts up to `chunk_size` files at a time and yields a state after each
        file. A failing file is recorded in `errors`; the others go on.

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.
            pdf_password: Password for encrypted PDF files.

        Yields:
            Conversion state updates.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        # Get and filter files
        # list_files returns absolute paths, so make the base comparable.
        base_dir = to_upath(directory).absolute()
        if not base_dir.exists():
            msg = f"Directory not found: {directory}"
            raise FileNotFoundError(msg)

        files = await list_files(
            base_dir,
            pattern=pattern,
            recursive=recursive,
            include_dirs=False,
            exclude=exclude,
            max_depth=max_depth,
        )
        supported_files: list[tuple[str, upath.UPath]] = []
        for file_path in files:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type in self.converter.SUPPORTED_MIME_TYPES:
                rel_path = str(file_path.relative_to(base_dir))
                supported_files.append((rel_path, file_path))

        results: dict[str, Document] = {}
        errors: dict[str, Exception] = {}
        semaphore = asyncio.Semaphore(self.chunk_size)

        async def convert(rel_path: str, path: upath.UPath) -> tuple[str, Document | Exception]:
            async with semaphore:
                try:
                    doc = await self.converter.convert_file(path, pdf_password=pdf_password)
                except Exception as e:  # noqa: BLE001
                    return rel_path, e
                return rel_path, doc

        tasks = [asyncio.ensure_future(convert(rel, path)) for rel, path in supported_files]
        try:
            for processed, next_done in enumerate(asyncio.as_completed(tasks), start=1):
                rel_path, outcome = await next_done
                if isinstance(outcome, Exception):
                    errors[rel_path] = outcome
                else:
                    results[rel_path] = outcome
                yield Conversion(
                    total_files=len(supported_files),
                    processed_files=processed,
                    current_file=rel_path,
                    successful_files=len(results),
                    failed_files=len(errors),
                    errors=errors,
                    results=results,
                )
        finally:
            for task in tasks:
                task.cancel()


if __name__ == "__main__":
    import asyncio

    from docler.converters.mistral_provider.provider import MistralConverter

    async def main() -> None:
        converter = MistralConverter()
        dir_converter = DirectoryConverter(converter)
        await dir_converter.convert("path/to/directory")

    asyncio.run(main())
