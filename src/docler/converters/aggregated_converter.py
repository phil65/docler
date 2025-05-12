"""Aggregated document converter that delegates to appropriate converters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyenv

from docler.configs.converter_configs import AggregatedConverterConfig
from docler.converters.base import DocumentConverter
from docler.converters.registry import ConverterRegistry


if TYPE_CHECKING:
    from mkdown import Document
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage


class AggregatedConverter(DocumentConverter[AggregatedConverterConfig]):
    """Converter that delegates to specialized converters based on mime type.

    This converter uses a registry of specialized converters and selects
    the most appropriate one for each file type. It acts as a single entry
    point that supports all file types supported by any registered converter.
    """

    Config = AggregatedConverterConfig
    NAME = "aggregated"

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: str | None = None,
        registry: ConverterRegistry | None = None,
    ):
        """Initialize the aggregated converter.

        Args:
            languages: Languages to use for conversion
            page_range: Page range to extract.
            registry: Existing registry to use, or create a new one if None
        """
        super().__init__(languages=languages, page_range=page_range)
        self._registry = registry or ConverterRegistry.create_default(languages=languages)

    @classmethod
    def from_config(cls, config: AggregatedConverterConfig) -> AggregatedConverter:
        """Create an AggregatedConverter instance from a configuration."""
        registry = ConverterRegistry()
        for converter_config in config.converters:
            converter = converter_config.get_provider()
            registry.register(converter)
        for mime_or_ext, converter_name in config.mime_preferences.items():
            registry.set_preference(mime_or_ext, converter_name)

        return cls(registry=registry)

    def to_config(self) -> AggregatedConverterConfig:
        """Extract configuration from the converter instance."""
        config = AggregatedConverterConfig()
        for converter in self._registry._converters:
            config.converters.append(converter.to_config())
        config.mime_preferences = dict(self._registry._preferences)
        return config

    def get_supported_mime_types(self) -> set[str]:
        """Get all MIME types supported by registered converters."""
        return self._registry.get_supported_mime_types()

    def set_converter_preference(self, mime_or_extension: str, converter_name: str):
        """Set a preference for which converter to use for a specific file type.

        Args:
            mime_or_extension: MIME type ('application/pdf') or file extension ('.pdf')
            converter_name: Name of the preferred converter ('mistral')
        """
        self._registry.set_preference(mime_or_extension, converter_name)

    async def _convert_path_async(
        self,
        file_path: StrPath,
        mime_type: MimeType,
    ) -> Document:
        """Delegate conversion to the appropriate converter."""
        converter = self._registry.get_converter(str(file_path), mime_type)
        if not converter:
            msg = f"No converter found for file: {file_path} (mime type: {mime_type})"
            raise ValueError(msg)

        try:
            return await converter._convert_path_async(file_path, mime_type)
        except NotImplementedError:
            return await anyenv.run_in_thread(
                converter._convert_path_sync, file_path, mime_type
            )
