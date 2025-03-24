"""Document converter using LlamaParse."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from docler.configs.converter_configs import LlamaParseConfig
from docler.converters.base import DocumentConverter
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = logging.getLogger(__name__)


class LlamaParseConverter(DocumentConverter[LlamaParseConfig]):
    """Document converter using LlamaParse."""

    Config = LlamaParseConfig

    NAME = "llamaparse"
    REQUIRED_PACKAGES: ClassVar = {"llama-parse"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Office Documents
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx  # noqa: E501
        "application/vnd.ms-word.document.macroEnabled.12",  # .docm
        "application/vnd.ms-powerpoint",  # .ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx # noqa: E501
        "application/vnd.ms-powerpoint.presentation.macroEnabled.12",  # .pptm
        "application/vnd.ms-excel",  # .xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel.sheet.macroEnabled.12",  # .xlsm
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12",  # .xlsb
        # Open/Libre Office
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
        # Text formats
        "text/html",
        "text/markdown",
        "text/plain",  # .txt
        "text/rtf",  # .rtf
        "text/csv",  # .csv
        "text/tab-separated-values",  # .tsv
        "application/xml",  # .xml
        "application/epub+zip",  # .epub
        # Images
        "image/jpeg",  # .jpg, .jpeg
        "image/png",  # .png
        "image/gif",  # .gif
        "image/bmp",  # .bmp
        "image/svg+xml",  # .svg
        "image/tiff",  # .tiff
        "image/webp",  # .webp
        # Audio
        "audio/mpeg",  # .mp3
        "audio/mp4",  # .mp4 audio
        "audio/wav",  # .wav
        "audio/webm",  # .webm audio
        "audio/m4a",  # .m4a
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        api_key: str | None = None,
    ):
        """Initialize the LlamaParse converter.

        Args:
            languages: List of supported languages
            api_key: LlamaParse API key, defaults to LLAMAPARSE_API_KEY env var
        """
        super().__init__(languages=languages)
        self.api_key = api_key or get_api_key("LLAMAPARSE_API_KEY")

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document using LlamaParse."""
        import base64

        from llama_parse import LlamaParse, ResultType
        import requests
        import upath

        path = upath.UPath(file_path)
        parser = LlamaParse(api_key=self.api_key, result_type=ResultType.MD)
        result = parser.get_json_result(str(path))

        pages = result[0]["pages"]  # First document's pages
        job_id = result[0]["job_id"]
        content_parts: list[str] = []
        images: list[Image] = []

        for page in pages:
            if page.get("md"):
                content_parts.append(page["md"])

            # Process images directly from the page
            for img in page.get("images", []):
                image_count = len(images)
                id_ = f"img-{image_count}"

                # Get image data directly from API
                asset_name = img["name"]
                asset_url = f"{parser.base_url}/api/parsing/job/{job_id}/result/image/{asset_name}"  # noqa: E501

                response = requests.get(
                    asset_url, headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()
                img_data = base64.b64encode(response.content).decode("utf-8")

                # Determine image type or default to png
                img_type = "png"  # Default image type
                if "." in asset_name:
                    extension = asset_name.split(".")[-1].lower()
                    if extension in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
                        img_type = extension

                filename = f"{id_}.{img_type}"
                mime = f"image/{img_type}"

                image = Image(id=id_, content=img_data, mime_type=mime, filename=filename)
                images.append(image)

        return Document(
            content="\n\n".join(content_parts),
            images=images,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
            page_count=len(pages),
        )


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = LlamaParseConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
