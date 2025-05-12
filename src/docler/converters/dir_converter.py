"""Directory conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import mimetypes
from typing import TYPE_CHECKING

from upathtools import list_files


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mkdown import Document

    from docler.common_types import StrPath
    from docler.converters.base import DocumentConverter


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

    def __init__(self, converter: DocumentConverter, *, chunk_size: int = 50):
        """Initialize the directory converter.

        Args:
            converter: Document converter to use.
            chunk_size: Number of files to convert in parallel.
        """
        self.converter = converter
        self.chunk_size = chunk_size

    async def convert(
        self,
        directory: StrPath,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Document]:
        """Convert all supported files in a directory.

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.

        Returns:
            Mapping of relative paths to converted documents.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        async for _state in self.convert_with_progress(
            directory,
            pattern=pattern,
            recursive=recursive,
            exclude=exclude,
            max_depth=max_depth,
        ):
            pass
        return _state.results

    async def convert_with_progress(
        self,
        directory: StrPath,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
    ) -> AsyncIterator[Conversion]:
        """Convert files with progress updates.

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.

        Yields:
            Conversion state updates.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        import upath

        # Get and filter files
        base_dir = upath.UPath(directory)
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

        for i in range(0, len(supported_files), self.chunk_size):
            chunk = supported_files[i : i + self.chunk_size]

            try:
                # Convert chunk
                documents = await self.converter.convert_files([
                    path for _, path in chunk
                ])

                # Store results
                for (rel_path, _), doc in zip(chunk, documents):
                    results[rel_path] = doc

            except Exception as e:  # noqa: BLE001
                # Store errors for all files in failed chunk
                for rel_path, _ in chunk:
                    errors[rel_path] = e
            yield Conversion(
                total_files=len(supported_files),
                processed_files=i + len(chunk),
                current_file=chunk[-1][0],  # last file in chunk
                successful_files=len(results),
                failed_files=len(errors),
                errors=errors,
                results=results,
            )


if __name__ == "__main__":
    import asyncio

    from docler.converters.mistral_provider.provider import MistralConverter

    async def main():
        converter = MistralConverter()
        dir_converter = DirectoryConverter(converter)
        await dir_converter.convert("path/to/directory")

    asyncio.run(main())
