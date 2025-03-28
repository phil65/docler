"""Registry for document converters."""

from __future__ import annotations

import logging
import mimetypes
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.common_types import SupportedLanguage
    from docler.converters.base import DocumentConverter


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
        cls, languages: list[SupportedLanguage] | None = None
    ) -> ConverterRegistry:
        """Create a registry with all available converters.

        Args:
            languages: Languages to use for converters

        Returns:
            Registry with all converters registered
        """
        import importlib.util

        # Import all converter classes
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

        # All converter classes
        converter_classes: list[type[DocumentConverter]] = [
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

        for converter_cls in converter_classes:
            has_requirements = all(
                importlib.util.find_spec(package.replace("-", "_"))
                for package in converter_cls.REQUIRED_PACKAGES
            )

            if has_requirements:
                try:
                    converter = converter_cls(languages=languages)
                    # Register the converter instance
                    registry.register(converter)
                except Exception:
                    logging.exception("Failed to initialize %s", converter_cls.__name__)
                    continue

        return registry

    def register(self, converter: DocumentConverter):
        """Register a converter.

        Args:
            converter: Converter instance to register.
        """
        self._converters.append(converter)

    def get_converter(
        self,
        file_path: str,
        mime_type: str | None = None,
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
                    and mime_type in converter.__class__.SUPPORTED_MIME_TYPES
                ):
                    return converter

        # No preference, use first converter that supports this MIME type
        for converter in self._converters:
            if mime_type in converter.__class__.SUPPORTED_MIME_TYPES:
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
        """Get all MIME types supported by registered converters.

        Returns:
            Set of supported MIME type strings
        """
        mime_types = set()
        for converter in self._converters:
            mime_types.update(converter.__class__.SUPPORTED_MIME_TYPES)
        return mime_types


if __name__ == "__main__":
    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    async def main():
        registry = ConverterRegistry.create_default(languages=["en"])
        converter = registry.get_converter("document.pdf")
        if converter:
            print(f"Found converter: {converter.NAME}")
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
