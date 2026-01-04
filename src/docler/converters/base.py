"""Base converter interface for document processing."""

from __future__ import annotations

from abc import ABC
import base64
from dataclasses import dataclass, field
from io import BytesIO
import mimetypes
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar

import anyenv
from pydantic import BaseModel
from upathtools import read_path, to_upath

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mkdown import Document, Image
    from schemez import MimeType
    import upath
    from upath.types import JoinablePathLike

    from docler.common_types import PageRangeString, SupportedLanguage
    from docler_config.converter_configs import ConverterConfig


@dataclass
class ConverterResult:
    """Intermediate result from document conversion.

    This is returned by the internal conversion methods before the base class
    assembles the final Document with source path and other metadata.
    """

    content: str
    """The converted markdown content."""
    images: list[Image] = field(default_factory=list)
    """Extracted images from the document."""
    title: str | None = None
    """Optional title. If None, base class derives from filename."""
    metadata: dict[str, Any] | None = None
    """Optional metadata from the conversion process."""


def _get_extension_for_mime(mime_type: str) -> str:
    """Get file extension for a MIME type."""
    ext = mimetypes.guess_extension(mime_type)
    return ext or ".bin"


class DocumentConverter[TConfig: BaseModel = Any](BaseProvider[TConfig], ABC):
    """Abstract base class for document converters.

    Implementation classes should override either:
    - _convert_sync: For CPU-bound operations
    - _convert_async: For IO-bound/API-based operations

    Both methods receive BytesIO and mime_type, returning a ConverterResult.
    The base class handles file I/O and assembles the final Document.
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
        page_range: PageRangeString | None = None,
    ) -> None:
        super().__init__()
        self.languages = languages
        self.page_range = page_range

    def __init_subclass__(cls, **kwargs: Any) -> None:
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

    async def convert_content(
        self,
        content: bytes | BytesIO | str,
        mime_type: MimeType,
        *,
        source_path: str | None = None,
        title: str | None = None,
        pdf_password: str | None = None,
    ) -> Document:
        """Convert document content directly from bytes, BytesIO, or base64 string.

        Args:
            content: Document content as bytes, BytesIO, or base64-encoded string.
            mime_type: MIME type of the content.
            source_path: Optional source path for metadata.
            title: Optional title. If None, uses "Untitled".
            pdf_password: Password for encrypted PDF files.

        Returns:
            Converted document with extracted text and images.

        Raises:
            ValueError: If the content type or MIME type is not supported.
        """
        from mkdown import Document

        if not self.supports_mime_type(mime_type):
            supported = self.get_supported_mime_types()
            msg = f"Unsupported MIME type {mime_type}. Must be one of: {supported}"
            raise ValueError(msg)

        # Handle different input types
        if isinstance(content, str):
            # Assume base64-encoded string
            try:
                content_bytes = base64.b64decode(content)
            except Exception as e:
                msg = f"Failed to decode base64 content: {e}"
                raise ValueError(msg) from e
            data = BytesIO(content_bytes)
        elif isinstance(content, bytes):
            data = BytesIO(content)
        elif isinstance(content, BytesIO):
            data = content
        else:
            msg = f"Unsupported content type: {type(content)}"
            raise TypeError(msg)

        # Handle PDF decryption if needed
        if mime_type == "application/pdf":
            from docler.pdf_utils import decrypt_pdf, get_pdf_info

            try:
                pdf_info = get_pdf_info(data.getvalue())
                if pdf_info.is_encrypted:
                    if pdf_password is None:
                        # Try empty password first
                        try:
                            decrypted_content = decrypt_pdf(data.getvalue(), None)
                            data = BytesIO(decrypted_content)
                        except ValueError as e:
                            if "requires a password" in str(e):
                                msg = (
                                    "PDF is encrypted but no password provided. "
                                    "Please provide pdf_password parameter."
                                )
                                raise ValueError(msg) from e
                            raise
                    else:
                        # Decrypt the PDF with provided password
                        try:
                            decrypted_content = decrypt_pdf(data.getvalue(), pdf_password)
                            data = BytesIO(decrypted_content)
                        except ValueError as e:
                            if "Incorrect password" in str(e):
                                raise ValueError("Incorrect PDF password") from e
                            raise
            except ValueError as e:
                if "encrypted" not in str(e).lower():
                    # If it's not an encryption issue, let it pass through to the converter
                    pass
                else:
                    raise

        result = await self._convert_threaded(data, mime_type)

        # Assemble final Document
        document = Document(
            content=result.content,
            images=result.images,
            title=result.title or title or "Untitled",
            source_path=source_path or "",
            mime_type=mime_type,
            metadata=result.metadata or {},
        )

        # Inject conversion cost into metadata
        if self.price_per_page is not None and document.page_count > 0:
            total_cost = self.price_per_page * document.page_count
            if document.metadata is None:
                document.metadata = {}
            document.metadata.update({
                "conversion_cost_usd": total_cost,
                "price_per_page_usd": self.price_per_page,
                "pages_processed": document.page_count,
            })

        return document

    async def convert_files(
        self, file_paths: Sequence[JoinablePathLike], *, pdf_password: str | None = None
    ) -> list[Document]:
        """Convert multiple document files in parallel.

        Args:
            file_paths: Sequence of paths to documents to convert.
            pdf_password: Password for encrypted PDF files.

        Returns:
            List of converted documents in the same order as the input paths.

        Raises:
            FileNotFoundError: If any file doesn't exist.
            ValueError: If any file format is not supported.
        """
        tasks = [self.convert_file(path, pdf_password=pdf_password) for path in file_paths]
        return list(await anyenv.gather(*tasks))

    async def convert_file(
        self, file_path: JoinablePathLike, *, pdf_password: str | None = None
    ) -> Document:
        """Convert a document file.

        Supports both local and remote files through fsspec/upath.

        Args:
            file_path: Path to the file to process (local or remote URI).
            pdf_password: Password for encrypted PDF files.

        Returns:
            Converted document with extracted text and images.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        path = to_upath(file_path)
        if not path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            msg = f"Could not determine mime type for: {file_path}"
            raise ValueError(msg)

        # Read file content
        if path.protocol in self.SUPPORTED_PROTOCOLS:
            content = await anyenv.run_in_thread(path.read_bytes)
        else:
            content = await read_path(path, mode="rb")

        # Use convert_content for the actual conversion
        return await self.convert_content(
            content=content,
            mime_type=mime_type,
            source_path=str(path),
            title=path.stem,
            pdf_password=pdf_password,
        )

    async def _convert_threaded(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Internal method to handle conversion routing.

        Will use _convert_async if implemented, otherwise falls back to
        running _convert_sync in a thread.
        """
        try:
            return await self._convert_async(data, mime_type)
        except NotImplementedError:
            return await anyenv.run_in_thread(self._convert_sync, data, mime_type)

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Synchronous implementation for CPU-bound operations.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file.

        Returns:
            Intermediate conversion result.
        """
        raise NotImplementedError

    async def _convert_async(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Asynchronous implementation for IO-bound operations.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file.

        Returns:
            Intermediate conversion result.
        """
        raise NotImplementedError

    def _write_temp_file(
        self, data: BytesIO, mime_type: MimeType
    ) -> tempfile._TemporaryFileWrapper[bytes]:
        """Write BytesIO to a temporary file for converters that need file paths.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type to determine file extension.

        Returns:
            NamedTemporaryFile object. Caller is responsible for cleanup.
        """
        ext = _get_extension_for_mime(mime_type)
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)  # noqa: SIM115
        tmp.write(data.read())
        tmp.flush()
        data.seek(0)  # Reset for potential reuse
        return tmp

    async def convert_directory(
        self,
        directory: JoinablePathLike,
        *,
        pattern: str = "**/*",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
        chunk_size: int = 50,
        pdf_password: str | None = None,
    ) -> dict[str, Document]:
        """Convert all supported files in a directory.

        Args:
            directory: Base directory to read from.
            pattern: Glob pattern to match files against.
            recursive: Whether to search subdirectories.
            exclude: List of patterns to exclude.
            max_depth: Maximum directory depth for recursive search.
            chunk_size: Number of files to convert in parallel.
            pdf_password: Password for encrypted PDF files.

        Returns:
            Mapping of relative paths to converted documents.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        import mimetypes

        from upathtools import list_files

        # Get directory listing
        base_dir = to_upath(directory)
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
            documents = await self.convert_files(
                [path for _, path in chunk], pdf_password=pdf_password
            )

            # Store results with relative paths as keys
            for (rel_path, _), doc in zip(chunk, documents):
                results[rel_path] = doc

        return results
