"""Registry for document converters."""

from __future__ import annotations

import mimetypes
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.base import DocumentConverter
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage
    from docler.models import Document


class ConverterRegistry:
    """Registry for document converters.

    Allows mapping mime types to converter implementations with priorities.
    Higher priority values mean higher precedence.
    """

    def __init__(self) -> None:
        """Initialize an empty converter registry."""
        # Dict[mime_type, List[Tuple[priority, converter_cls]]]
        self._converters: dict[str, list[tuple[int, type[DocumentConverter]]]] = {}

    @classmethod
    def create_default(cls):
        from docler.marker_provider import MarkerConverter
        from docler.mistral_provider import MistralConverter

        registry = ConverterRegistry()

        # Register converters with priorities
        # Base priority for most formats
        registry.register(MarkerConverter, priority=0)
        # Prefer Mistral for PDFs
        registry.register(MistralConverter, ["application/pdf"], priority=100)
        return registry

    def register(
        self,
        converter_cls: type[DocumentConverter],
        mime_types: list[str] | None = None,
        *,
        priority: int = 0,
    ) -> None:
        """Register a converter for specific mime types.

        Args:
            converter_cls: Converter class to register.
            mime_types: List of mime types this converter should handle.
                If None, uses the converter's SUPPORTED_MIME_TYPES.
            priority: Priority of this converter (higher = more preferred).
        """
        types_to_register = mime_types or list(converter_cls.SUPPORTED_MIME_TYPES)

        for mime_type in types_to_register:
            if mime_type not in self._converters:
                self._converters[mime_type] = []

            # Add converter with priority and sort by priority (highest first)
            self._converters[mime_type].append((priority, converter_cls))
            self._converters[mime_type].sort(reverse=True)

    def get_converter(
        self,
        file_path: str,
    ) -> type[DocumentConverter] | None:
        """Get the highest priority converter for a file.

        Args:
            file_path: Path to the file to convert.

        Returns:
            Highest priority converter class for this file type,
            or None if no converter is registered.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            return None

        converters = self._converters.get(mime_type, [])
        return converters[0][1] if converters else None

    async def convert_file(
        self,
        file_path: StrPath,
        language: SupportedLanguage,
        **converter_kwargs,
    ) -> Document:
        """Convert a single file using the appropriate converter.

        Args:
            file_path: Path to the file to convert
            language: Primary language for OCR/processing
            **converter_kwargs: Additional arguments to pass to the converter

        Returns:
            Converted document

        Raises:
            ValueError: If no suitable converter is found
        """
        converter_cls = self.get_converter(str(file_path))
        if not converter_cls:
            msg = f"No converter found for file: {file_path}"
            raise ValueError(msg)

        converter = converter_cls(languages=[language], **converter_kwargs)
        return await converter.convert_file(file_path)

    # async def convert_directory(
    #     self,
    #     directory: StrPath,
    #     language: SupportedLanguage,
    #     *,
    #     pattern: str = "**/*",
    #     recursive: bool = True,
    #     exclude: list[str] | None = None,
    #     max_depth: int | None = None,
    #     chunk_size: int = 50,
    #     **converter_kwargs,
    # ) -> dict[str, Document]:
    #     """Convert all supported files in a directory.

    #     Args:
    #         directory: Directory to process
    #         language: Primary language for OCR/processing
    #         pattern: File glob pattern
    #         recursive: Whether to process subdirectories
    #         exclude: Patterns to exclude
    #         max_depth: Maximum directory depth
    #         chunk_size: Files to process in parallel
    #         **converter_kwargs: Additional arguments for converters

    #     Returns:
    #         Map of relative paths to converted documents
    #     """
    #     from docler.dir_converter import DirectoryConverter

    #     # Find the first matching converter for any file
    #     files = await DirectoryConverter._list_files(
    #         directory, pattern, recursive, exclude, max_depth
    #     )
    #     for file in files:
    #         if converter_cls := self.get_converter(str(file)):
    #             converter = converter_cls(languages=[language], **converter_kwargs)
    #             dir_converter = DirectoryConverter(converter, chunk_size=chunk_size)
    #             return await dir_converter.convert(
    #                 directory,
    #                 pattern=pattern,
    #                 recursive=recursive,
    #                 exclude=exclude,
    #                 max_depth=max_depth,
    #             )

    #     msg = f"No suitable converter found for any files in {directory}"
    #     raise ValueError(msg)

    # async def convert_directory_with_progress(
    #     self,
    #     directory: StrPath,
    #     language: SupportedLanguage,
    #     *,
    #     pattern: str = "**/*",
    #     recursive: bool = True,
    #     exclude: list[str] | None = None,
    #     max_depth: int | None = None,
    #     chunk_size: int = 50,
    #     **converter_kwargs,
    # ) -> AsyncIterator[Conversion]:
    #     """Convert directory with progress updates.

    #     Args:
    #         directory: Directory to process
    #         language: Primary language for OCR/processing
    #         pattern: File glob pattern
    #         recursive: Whether to process subdirectories
    #         exclude: Patterns to exclude
    #         max_depth: Maximum directory depth
    #         chunk_size: Files to process in parallel
    #         **converter_kwargs: Additional arguments for converters

    #     Yields:
    #         Conversion progress states
    #     """
    #     from docler.dir_converter import DirectoryConverter

    #     # Find the first matching converter for any file
    #     files = await DirectoryConverter._list_files(
    #         directory, pattern, recursive, exclude, max_depth
    #     )
    #     for file in files:
    #         if converter_cls := self.get_converter(str(file)):
    #             converter = converter_cls(languages=[language], **converter_kwargs)
    #             dir_converter = DirectoryConverter(converter, chunk_size=chunk_size)
    #             async for state in dir_converter.convert_with_progress(
    #                 directory,
    #                 pattern=pattern,
    #                 recursive=recursive,
    #                 exclude=exclude,
    #                 max_depth=max_depth,
    #             ):
    #                 yield state
    #             return

    #     msg = f"No suitable converter found for any files in {directory}"
    #     raise ValueError(msg)


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    async def main():
        registry = ConverterRegistry.create_default()
        pdf_path = "document.pdf"
        return await registry.convert_file(pdf_path, language="en")

    result = anyenv.run_sync(main())
    print(result)
