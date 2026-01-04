"""Azure Document Intelligence converter implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from docler.converters.azure_provider.utils import (
    get_metadata,
    replace_page_breaks,
    to_image,
    update_content,
)
from docler.converters.base import ConverterResult, DocumentConverter
from docler.converters.exceptions import MissingConfigurationError
from docler.log import get_logger
from docler.utils import get_api_key
from docler_config.converter_configs import AzureConfig


if TYPE_CHECKING:
    from io import BytesIO

    from azure.ai.documentintelligence.models import AnalyzeResult
    from mkdown import Image
    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage
    from docler_config.converter_configs import AzureFeatureFlag, AzureModel

logger = get_logger(__name__)

ENV_ENDPOINT = "AZURE_DOC_INTELLIGENCE_ENDPOINT"
ENV_API_KEY = "AZURE_DOC_INTELLIGENCE_KEY"


class AzureConverter(DocumentConverter[AzureConfig]):
    """Document converter using Azure Document Intelligence."""

    Config = AzureConfig

    NAME = "azure"
    REQUIRED_PACKAGES: ClassVar = {"azure"}
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
        page_range: PageRangeString | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: AzureModel = "prebuilt-layout",
        additional_features: set[AzureFeatureFlag] | None = None,
    ):
        """Initialize Azure Document Intelligence converter.

        Args:
            languages: ISO language codes for OCR, defaults to ['en']
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
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

        super().__init__(languages=languages, page_range=page_range)

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

    @property
    def price_per_page(self) -> float:
        """Price per page in USD."""
        return 0.00958

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
                image = to_image(response_iter, i)
                images.append(image)

        return images

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Convert a document file synchronously using Azure Document Intelligence.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file.

        Returns:
            Intermediate conversion result.
        """
        from azure.ai.documentintelligence.models import (
            AnalyzeOutputOption,
            DocumentAnalysisFeature,
        )
        from azure.core.exceptions import HttpResponseError

        features = [getattr(DocumentAnalysisFeature, f) for f in self.features]
        try:
            poller = self._client.begin_analyze_document(
                model_id=self.model,
                body=data,
                pages=self.page_range,
                features=features,
                output=[AnalyzeOutputOption.FIGURES],
                locale=self.languages[0] if self.languages else None,
                output_content_format="markdown",
            )
            result: AnalyzeResult = poller.result()
            content = result.content
        except HttpResponseError as e:
            msg = f"Azure Document Intelligence failed: {e.message}"
            if e.error:
                msg = f"{msg} (Error code: {e.error.code})"
            raise ValueError(msg) from e
        else:
            metadata = get_metadata(result)
            images = self._convert_azure_images(result, poller.details["operation_id"])
            content = replace_page_breaks(content)

            if images:
                content = update_content(content, images)
            return ConverterResult(content=content, images=images, metadata=metadata)


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = AzureConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
    print(result.images)
