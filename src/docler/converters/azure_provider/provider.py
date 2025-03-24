"""Azure Document Intelligence converter implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from docler.configs.converter_configs import AzureConfig
from docler.converters.base import DocumentConverter
from docler.converters.exceptions import MissingConfigurationError
from docler.models import Document, Image
from docler.utils import get_api_key


if TYPE_CHECKING:
    from collections.abc import Sequence

    from azure.ai.documentintelligence.models import AnalyzeResult

    from docler.common_types import StrPath, SupportedLanguage

logger = logging.getLogger(__name__)

PrebuiltModel = Literal[
    "prebuilt-read",
    "prebuilt-layout",
    "prebuilt-idDocument",
    "prebuilt-receipt",
]

OcrFeatureFlag = Literal[
    "ocrHighResolution",
    "languages",
    "barcodes",
    "formulas",
    "keyValuePairs",
    "styleFont",
    "queryFields",
]
ENV_ENDPOINT = "AZURE_DOC_INTELLIGENCE_ENDPOINT"
ENV_API_KEY = "AZURE_DOC_INTELLIGENCE_KEY"


class AzureConverter(DocumentConverter[AzureConfig]):
    """Document converter using Azure Document Intelligence."""

    Config = AzureConfig

    NAME = "azure"
    REQUIRED_PACKAGES: ClassVar = {""}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDF
        "application/pdf",
        # Office Documents
        "application/msword",  # doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.ms-powerpoint",  # ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx  # noqa: E501
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
        # Images
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: PrebuiltModel = "prebuilt-layout",
        additional_features: Sequence[OcrFeatureFlag] | None = None,
    ):
        """Initialize Azure Document Intelligence converter.

        Args:
            languages: ISO language codes for OCR, defaults to ['en']
            endpoint: Azure service endpoint URL. Falls back to env var.
            api_key: Azure API key. Falls back to env var.
            model: Pre-trained model to use
            additional_features: Optional add-on capabilities like
                BARCODES, FORMULAS, OCR_HIGH_RESOLUTION etc.

        Raises:
            MissingConfigurationError: If endpoint or API key cannot be found
        """
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        super().__init__(languages=languages)

        self.endpoint = endpoint or get_api_key(ENV_ENDPOINT)
        self.api_key = api_key or get_api_key(ENV_API_KEY)

        self.model = model
        self.features = list(additional_features) if additional_features else []

        try:
            credential = AzureKeyCredential(self.api_key)
            self._client = DocumentIntelligenceClient(self.endpoint, credential)
        except Exception as e:
            msg = "Failed to create Azure client"
            raise MissingConfigurationError(msg) from e

    def _convert_azure_images(
        self,
        result: AnalyzeResult,
        operation_id: str,
    ) -> list[Image]:
        """Extract and convert images from Azure results.

        Args:
            result: Azure document analysis result
            operation_id: Azure operation ID for retrieving figures

        Returns:
            List of extracted images
        """
        images: list[Image] = []
        if result.figures:
            for i, figure in enumerate(result.figures):
                if not figure.id:
                    continue
                response_iter = self._client.get_analyze_result_figure(
                    model_id=result.model_id,
                    result_id=operation_id,
                    figure_id=figure.id,
                )
                content = b"".join(response_iter)
                image_id = f"img-{i}"
                filename = f"{image_id}.png"
                image = Image(
                    id=image_id,
                    content=content,
                    mime_type="image/png",
                    filename=filename,
                )
                images.append(image)

        return images

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document file synchronously using Azure Document Intelligence."""
        import re

        from azure.ai.documentintelligence.models import (
            AnalyzeOutputOption,
            DocumentAnalysisFeature,
        )
        from azure.core.exceptions import HttpResponseError
        import upath

        path = upath.UPath(file_path)
        features = [
            getattr(DocumentAnalysisFeature, feature) for feature in self.features
        ]

        try:
            with path.open("rb") as f:
                poller = self._client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=f,
                    features=features,
                    output=[AnalyzeOutputOption.FIGURES],
                    locale=self.languages[0] if self.languages else "en",
                    output_content_format="markdown",
                )
            result = poller.result()
            operation_id = poller.details["operation_id"]

            metadata = {}
            if result.documents:
                doc = result.documents[0]  # Get first document
                if doc.fields:
                    metadata = {
                        name: field.get("valueString") or field.get("content", "")
                        for name, field in doc.fields.items()
                    }

            images = self._convert_azure_images(result, operation_id)
            # Process content to replace <figure> tags with markdown image references
            content = result.content
            if images:
                figure_pattern = r"<figure>(.*?)</figure>"
                figure_blocks = re.findall(figure_pattern, content, re.DOTALL)
                for i, block in enumerate(figure_blocks):
                    if i < len(images):
                        image = images[i]
                        img_ref = f"\n\n![{image.id}]({image.filename})\n\n"
                        content = content.replace(f"<figure>{block}</figure>", img_ref, 1)

            return Document(
                content=content,
                images=images,
                title=path.stem,
                source_path=str(path),
                mime_type=mime_type,
                page_count=len(result.pages) if result.pages else None,
                **metadata,
            )

        except HttpResponseError as e:
            msg = f"Azure Document Intelligence failed: {e.message}"
            if e.error:
                msg = f"{msg} (Error code: {e.error.code})"
            raise ValueError(msg) from e


if __name__ == "__main__":
    import anyenv

    logging.basicConfig(level=logging.DEBUG)
    pdf_path = "src/docler/resources/pdf_sample.pdf"

    converter = AzureConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
    print(result.images)
