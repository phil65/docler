"""Base converter interface for document processing."""

from __future__ import annotations

from abc import ABC
import mimetypes
import tempfile
from typing import TYPE_CHECKING, ClassVar

import anyenv
from upathtools import read_path

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mkdown import Document
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage
    from docler.configs.converter_configs import ConverterConfig


class DocumentConverter[TConfig](BaseProvider[TConfig], ABC):
    """Abstract base class for document converters.

    Implementation classes should override either:
    - _convert_path_sync: For CPU-bound operations
    - _convert_path_async: For IO-bound/API-based operations
    """

    Config: ClassVar[type[ConverterConfig]]

    NAME: str
    """Name of the converter."""
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = set()
    """Mime types this converter can handle."""
    SUPPORTED_PROTOCOLS: ClassVar[set[str]] = {"file", ""}
    """Protocols this converter can handle.

    Non-supported protocols will get handled using fsspec + a temporary file.
    """
    registry: ClassVar[dict[str, type[DocumentConverter]]] = {}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        page_range: str | None = None,
    ):
        super().__init__()
        self.languages = languages
        self.page_range = page_range

    def __init_subclass__(cls, **kwargs):
        """Register subclasses automatically when they're defined."""
        super().__init_subclass__(**kwargs)
        DocumentConverter.registry[cls.NAME] = cls

    @property
    def price_per_page(self) -> float | None:
        """Price per page in USD."""
        return None

    def get_supported_mime_types(self) -> set[str]:
        """Get all MIME types supported by this converter.

        Returns:
            Set of supported MIME type strings
        """
        return self.SUPPORTED_MIME_TYPES

    def supports_mime_type(self, mime_type: MimeType) -> bool:
        """Check if this converter supports a specific MIME type.

        Args:
            mime_type: MIME type to check

        Returns:
            True if this converter supports the MIME type
        """
        return mime_type in self.get_supported_mime_types()

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

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            msg = f"Could not determine mime type for: {file_path}"
            raise ValueError(msg)
        if mime_type not in self.SUPPORTED_MIME_TYPES:
            msg = (
                f"Unsupported file type {mime_type}."
                f"Must be one of: {self.SUPPORTED_MIME_TYPES}"
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
        mime_type: MimeType,
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

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Synchronous implementation for CPU-bound operations."""
        raise NotImplementedError

    async def _convert_path_async(
        self,
        file_path: StrPath,
        mime_type: MimeType,
    ) -> Document:
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
