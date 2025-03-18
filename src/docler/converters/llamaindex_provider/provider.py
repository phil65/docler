"""Document converter using LlamaIndex's SmartPDFLoader."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, ClassVar

from docler.converters.base import DocumentConverter
from docler.models import Document, Image


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.converters.base import SupportedLanguage


logger = logging.getLogger(__name__)
DEFAULT_URL = (
    "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
)


class LlamaIndexConverter(DocumentConverter):
    """Document converter using LlamaIndex's SmartPDFLoader."""

    NAME = "llamaindex"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}
    SUPPORTED_PROTOCOLS: ClassVar[set[str]] = {"", "file", "http", "https"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        llmsherpa_api_url: str = DEFAULT_URL,
        **parser_kwargs,
    ):
        """Initialize the LlamaIndex converter.

        Args:
            languages: List of supported languages
            llmsherpa_api_url: API endpoint for LLMSherpa PDF parser
            **parser_kwargs: Additional arguments for SmartPDFLoader
        """
        super().__init__(languages=languages)
        self.api_url = llmsherpa_api_url
        self.parser_kwargs = parser_kwargs

    async def _convert_path_async(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a file using LlamaIndex's SmartPDFLoader.

        Args:
            file_path: Path or URL to the PDF file
            mime_type: MIME type of the file (must be PDF)

        Returns:
            Converted document with extracted text and images

        Raises:
            ImportError: If llama_index isn't installed
            ValueError: If conversion fails
        """
        from llama_index.readers.smart_pdf_loader import SmartPDFLoader
        import upath

        path = upath.UPath(file_path)

        try:
            loader = SmartPDFLoader(llmsherpa_api_url=self.api_url, **self.parser_kwargs)
            llama_docs = loader.load_data(str(path))
            # Combine text from all pages/documents
            all_text_parts: list[str] = []
            all_images: list[Image] = []
            metadata = {}

            for i, doc in enumerate(llama_docs):
                # Add text content
                if doc.text:
                    all_text_parts.append(doc.text)

                # Process images if present
                if "images" in doc.metadata:
                    for img_data in doc.metadata["images"]:
                        if not img_data.get("image_base64"):
                            continue

                        # Create standardized image ID
                        image_count = len(all_images)
                        image_id = f"img-{image_count}"

                        # Get base64 data and remove prefix if present
                        img_content = img_data["image_base64"]
                        if "," in img_content:
                            img_content = img_content.split(",", 1)[1]

                        image = Image(
                            id=image_id,
                            content=base64.b64decode(img_content),
                            mime_type="image/png",
                            filename=f"{image_id}.png",
                        )
                        all_images.append(image)
                if i == 0:
                    metadata = {
                        k: v
                        for k, v in doc.metadata.items()
                        if k not in ("images", "page_number")
                    }

            return Document(
                content="\n\n".join(all_text_parts),
                images=all_images,
                title=metadata.get("title") or path.stem,
                author=metadata.get("author"),
                source_path=str(path),
                mime_type=mime_type,
                page_count=len(llama_docs),
                **metadata,
            )

        except Exception as e:
            msg = f"LlamaIndex conversion failed: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "https://arxiv.org/pdf/1910.13461.pdf"
    converter = LlamaIndexConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
