"""Registry for document converters."""

from __future__ import annotations

import mimetypes
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docler.base import DocumentConverter


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


if __name__ == "__main__":

    async def main():
        registry = ConverterRegistry.create_default()
        file_path = "document.pdf"
        converter_cls = registry.get_converter(file_path)
        assert converter_cls is not None
        converter = converter_cls()
        return await converter.convert_file(file_path)

    import anyenv

    result = anyenv.run_sync(main())
    print(result)
