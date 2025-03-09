"""Base converter interface for document processing."""

from __future__ import annotations

from abc import ABC
import mimetypes
import tempfile
from typing import TYPE_CHECKING, ClassVar

import anyenv
from upathtools import read_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage

    from .models import Document


class DocumentConverter(ABC):
    """Abstract base class for document converters.

    Implementation classes should override either:
    - _convert_path_sync: For CPU-bound operations
    - _convert_path_async: For IO-bound/API-based operations
    """

    NAME: str
    """Name of the converter."""
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = set()
    """Mime types this converter can handle."""
    SUPPORTED_PROTOCOLS: ClassVar[set[str]] = {"file", ""}
    """Protocols this converter can handle.

    Non-supported protocols will get handled using fsspec + a temporary file.
    """

    def __init__(self, languages: list[SupportedLanguage] | None = None) -> None:
        self.languages = languages

    async def convert_files(self, file_paths: Sequence[StrPath]) -> list[Document]:
        """Convert multiple document files in parallel.

        Args:
            file_paths: Sequence of paths to documents to convert.

        Returns:
            List of converted documents in the same order as the input paths.

        Raises:
            FileNotFoundError: If any file doesn't exist.
            ValueError: If any file format is not supported.
        """
        tasks = [self.convert_file(path) for path in file_paths]
        return await anyenv.gather(*tasks)  # type: ignore

    async def convert_file(self, file_path: StrPath) -> Document:
        """Convert a document file using Marker.

        Supports both local and remote files through fsspec/upath.

        Args:
            file_path: Path to the file to process (local or remote URI).

        Returns:
            Converted document with extracted text and images.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        import upath

        path = upath.UPath(file_path)
        if not path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            msg = f"Could not determine mime type for: {file_path}"
            raise ValueError(msg)
        if mime_type not in self.SUPPORTED_MIME_TYPES:
            msg = (
                f"Unsupported file type {mime_type}."
                "Must be one of: {self.SUPPORTED_MIME_TYPES}"
            )
            raise ValueError(msg)

        # For local files, convert directly
        if path.protocol in self.SUPPORTED_PROTOCOLS:
            return await self._convert_path_threaded(path, mime_type)

        # For remote files, download to temporary file first
        content = await read_path(path, mode="rb")
        with tempfile.NamedTemporaryFile(suffix=path.suffix) as temp_file:
            temp_path = upath.UPath(temp_file.name)
            temp_path.write_bytes(content)
            return await self._convert_path_threaded(temp_path, mime_type)

    async def _convert_path_threaded(
        self,
        file_path: StrPath,
        mime_type: str,
    ) -> Document:
        """Internal method to handle conversion routing.

        Will use _convert_path_async if implemented, otherwise falls back to
        running _convert_path_sync in a thread.
        """
        try:
            return await self._convert_path_async(file_path, mime_type)
        except NotImplementedError:
            return await anyenv.run_in_thread(
                self._convert_path_sync, file_path, mime_type
            )

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Synchronous implementation for CPU-bound operations."""
        raise NotImplementedError

    async def _convert_path_async(self, file_path: StrPath, mime_type: str) -> Document:
        """Asynchronous implementation for IO-bound operations."""
        raise NotImplementedError

    async def convert_directory(
        self,
        directory: StrPath,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
        chunk_size: int = 50,
    ) -> dict[str, Document]:
        """Convert all supported files in a directory.

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.
            chunk_size: Number of files to convert in parallel.

        Returns:
            Mapping of relative paths to converted documents.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        import mimetypes

        import upath
        from upathtools import list_files

        # Get directory listing
        base_dir = upath.UPath(directory)
        if not base_dir.exists():
            msg = f"Directory not found: {directory}"
            raise FileNotFoundError(msg)

        # Get all matching files
        files = await list_files(
            base_dir,
            pattern=pattern,
            recursive=recursive,
            include_dirs=False,
            exclude=exclude,
            max_depth=max_depth,
        )

        # Filter for supported mime types
        supported_files: list[tuple[str, upath.UPath]] = []
        for file_path in files:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type in self.SUPPORTED_MIME_TYPES:
                # Store both relative path and full path
                rel_path = str(file_path.relative_to(base_dir))
                supported_files.append((rel_path, file_path))

        # Convert files in chunks
        results: dict[str, Document] = {}
        for i in range(0, len(supported_files), chunk_size):
            chunk = supported_files[i : i + chunk_size]
            # Convert using full paths
            documents = await self.convert_files([path for _, path in chunk])

            # Store results with relative paths as keys
            for (rel_path, _), doc in zip(chunk, documents):
                results[rel_path] = doc

        return results
