"""Registry for document converters."""

from __future__ import annotations

import logging
import mimetypes
from typing import TYPE_CHECKING, Any

from docler.converters.base import DocumentConverter


if TYPE_CHECKING:
    from schemez import MimeType

    from docler.common_types import SupportedLanguage


class ConverterRegistry:
    """Registry for document converters."""

    def __init__(self):
        """Initialize an empty converter registry."""
        # All registered converters
        self._converters: list[DocumentConverter] = []
        # Preference overrides: {mime_type: converter_name}
        self._preferences: dict[str, str] = {}

    @classmethod
    def create_default(
        cls,
        languages: list[SupportedLanguage] | None = None,
    ) -> ConverterRegistry:
        """Create a registry with all available converters.

        Args:
            languages: Languages to use for converters
        """
        registry = cls()
        for converter_cls in DocumentConverter[Any].get_available_providers():
            if converter_cls.NAME == "aggregated":
                continue
            try:
                converter = converter_cls(languages=languages)
                registry.register(converter)
            except Exception:
                logging.exception("Failed to initialize %s", converter_cls.__name__)
                continue

        return registry

    def get_converter_by_name(self, name: str):
        return next(i for i in self._converters if name == i.NAME)

    def register(self, converter: DocumentConverter):
        """Register a converter."""
        self._converters.append(converter)

    def get_converter(
        self,
        file_path: str,
        mime_type: MimeType | None = None,
    ) -> DocumentConverter | None:
        """Get the appropriate converter for a file.

        Args:
            file_path: Path to the file to convert.
            mime_type: Optional explicit MIME type

        Returns:
            Converter instance for this file type, or None if no converter is registered.
        """
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                return None

        if mime_type in self._preferences:
            preferred_name = self._preferences[mime_type]
            for converter in self._converters:
                if (
                    preferred_name == converter.NAME
                    and mime_type in converter.get_supported_mime_types()
                ):
                    return converter

        # No preference, use first converter that supports this MIME type
        for converter in self._converters:
            if mime_type in converter.get_supported_mime_types():
                return converter

        return None

    def set_preference(self, mime_or_extension: str, converter_name: str):
        """Set a preference for a specific converter for a MIME type or file extension.

        Args:
            mime_or_extension: MIME type ('application/pdf') or file extension ('.pdf')
            converter_name: Name of the preferred converter
        """
        if "/" not in mime_or_extension:
            if not mime_or_extension.startswith("."):
                mime_or_extension = f".{mime_or_extension}"
            mime_type, _ = mimetypes.guess_type(f"dummy{mime_or_extension}")
            if mime_type:
                mime_or_extension = mime_type
        self._preferences[mime_or_extension] = converter_name

    def get_supported_mime_types(self) -> set[str]:
        """Get all MIME types supported by registered converters."""
        mime_types = set()
        for converter in self._converters:
            mime_types.update(converter.get_supported_mime_types())
        return mime_types


if __name__ == "__main__":
    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    async def main():
        registry = ConverterRegistry.create_default(languages=["en"])
        converter = registry.get_converter("document.pdf")
        if converter:
            print(f"Found converter: {converter.NAME}")
            #     pdf_path = "document.pdf"
            #     result = await converter.convert_file(pdf_path)
            #     print(f"Conversion successful: {len(result.content)} characters")
            #     return result

        else:
            print("No suitable converter found")

    result = anyenv.run_sync(main())
    print(result)
