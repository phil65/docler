"""Registry for document converters."""

from __future__ import annotations

import logging
import mimetypes
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage
    from docler.converters.base import DocumentConverter


class ConverterRegistry:
    """Registry for document converters.

    Allows mapping mime types to converter implementations with priorities.
    Higher priority values mean higher precedence.
    """

    def __init__(self):
        """Initialize an empty converter registry."""
        self._converters: dict[str, list[tuple[int, type[DocumentConverter]]]] = {}
        self._preferences: dict[str, str] = {}  # MIME type -> converter name preferences

    @classmethod
    def create_default(cls) -> ConverterRegistry:
        """Create a registry with all available converters.

        Returns:
            Registry with all converters registered with sensible priorities
        """
        import importlib.util

        from docler.converters.azure_provider import AzureConverter
        from docler.converters.datalab_provider import DataLabConverter
        from docler.converters.docling_provider import DoclingConverter
        from docler.converters.kreuzberg_provider import KreuzbergConverter
        from docler.converters.llamaparse_provider import LlamaParseConverter
        from docler.converters.llm_provider import LLMConverter
        from docler.converters.marker_provider import MarkerConverter
        from docler.converters.markitdown_provider import MarkItDownConverter
        from docler.converters.mistral_provider import MistralConverter
        from docler.converters.upstage_provider import UpstageConverter

        registry = cls()
        converters: list[type[DocumentConverter]] = [
            MarkerConverter,
            KreuzbergConverter,
            MarkItDownConverter,
            DoclingConverter,
            LLMConverter,
            DataLabConverter,
            AzureConverter,
            UpstageConverter,
            MistralConverter,
            LlamaParseConverter,
        ]

        default_priority = 10
        for converter_cls in converters:
            has_requirements = all(
                importlib.util.find_spec(package.replace("-", "_"))
                for package in converter_cls.REQUIRED_PACKAGES
            )
            if has_requirements:
                registry.register(converter_cls, priority=default_priority)
        return registry

    def register(
        self,
        converter_cls: type[DocumentConverter],
        mime_types: list[str] | None = None,
        *,
        priority: int = 0,
    ):
        """Register a converter for specific mime types.

        Args:
            converter_cls: Converter class to register.
            mime_types: List of mime types this converter should handle.
                If None, uses the converter's SUPPORTED_MIME_TYPES.
            priority: Priority of this converter (higher = more preferred).
        """
        # Use the class's SUPPORTED_MIME_TYPES attribute
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
        mime_type: str | None = None,
    ) -> type[DocumentConverter] | None:
        """Get the highest priority converter for a file.

        Args:
            file_path: Path to the file to convert.
            mime_type: Optional explicit MIME type

        Returns:
            Highest priority converter class for this file type,
            or None if no converter is registered.
        """
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                return None

        # Check if we have a preference for this MIME type
        if mime_type in self._preferences:
            preferred_name = self._preferences[mime_type]
            # Look for the preferred converter in all registered converters
            for converters in self._converters.values():
                for _, converter_cls in converters:
                    if (
                        preferred_name == converter_cls.NAME
                        and mime_type in converter_cls.SUPPORTED_MIME_TYPES
                    ):
                        return converter_cls

        # No preference or preferred converter not found, use highest priority
        converters = self._converters.get(mime_type, [])
        return converters[0][1] if converters else None

    def set_preference(self, mime_or_extension: str, converter_name: str):
        """Set a preference for a specific converter for a MIME type or file extension.

        Args:
            mime_or_extension: MIME type ('application/pdf') or file extension ('.pdf')
            converter_name: Name of the preferred converter
        """
        # Determine if this is a MIME type or file extension
        if "/" in mime_or_extension:
            # This is a MIME type
            self._preferences[mime_or_extension] = converter_name
        else:
            # This is a file extension - normalize it and get the MIME type
            if not mime_or_extension.startswith("."):
                mime_or_extension = f".{mime_or_extension}"

            mime_type, _ = mimetypes.guess_type(f"dummy{mime_or_extension}")
            if mime_type:
                self._preferences[mime_type] = converter_name

    def get_supported_mime_types(self) -> set[str]:
        """Get all MIME types supported by registered converters.

        Returns:
            Set of supported MIME type strings
        """
        return set(self._converters.keys())

    def create_converter(
        self, file_path: str, languages: list[SupportedLanguage] | None = None, **kwargs
    ) -> DocumentConverter | None:
        """Create appropriate converter instance for a file.

        Args:
            file_path: Path to the file to convert
            languages: Languages to use for conversion
            **kwargs: Additional arguments to pass to the converter

        Returns:
            Instantiated converter or None if no suitable converter found
        """
        converter_cls = self.get_converter(file_path)
        if not converter_cls:
            return None

        return converter_cls(languages=languages, **kwargs)


if __name__ == "__main__":
    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    async def main():
        registry = ConverterRegistry.create_default()
        converter_cls = registry.get_converter("document.pdf")
        if converter_cls:
            print(f"Found converter: {converter_cls.NAME}")
            converter = converter_cls(languages=["en"])
            try:
                pdf_path = "document.pdf"
                result = await converter.convert_file(pdf_path)
                print(f"Conversion successful: {len(result.content)} characters")
            except Exception as e:  # noqa: BLE001
                print(f"Conversion failed: {e}")
            else:
                return result

        else:
            print("No suitable converter found")
        return None

    result = anyenv.run_sync(main())
    print(result)
