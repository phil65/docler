"""Azure Document Intelligence converter implementation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar, Literal

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from docler.base import DocumentConverter
from docler.models import Document, Image


if TYPE_CHECKING:
    from collections.abc import Sequence

    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage

logger = logging.getLogger(__name__)

# Supported pre-trained models
PrebuiltModel = Literal[
    "prebuilt-read",
    "prebuilt-layout",
    "prebuilt-document",
    "prebuilt-idDocument",
    "prebuilt-receipt",
]

# Environment variable names
ENV_ENDPOINT = "AZURE_DOC_INTELLIGENCE_ENDPOINT"
ENV_API_KEY = "AZURE_DOC_INTELLIGENCE_KEY"


class AzureConverterError(Exception):
    """Base exception for Azure converter errors."""


class MissingConfigurationError(AzureConverterError):
    """Required Azure configuration is missing."""


class AzureConverter(DocumentConverter):
    """Document converter using Azure Document Intelligence."""

    NAME = "azure"
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
        model: PrebuiltModel = "prebuilt-document",
        additional_features: Sequence[str] | None = None,
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
        super().__init__(languages=languages)

        # Get configuration
        self.endpoint = endpoint or os.getenv(ENV_ENDPOINT)
        self.api_key = api_key or os.getenv(ENV_API_KEY)

        if not self.endpoint:
            msg = f"Azure endpoint not provided and {ENV_ENDPOINT} env var not set"
            raise MissingConfigurationError(msg)
        if not self.api_key:
            msg = f"Azure API key not provided and {ENV_API_KEY} env var not set"
            raise MissingConfigurationError(msg)

        self.model = model
        self.features = list(additional_features) if additional_features else []

        # Create client
        try:
            credential = AzureKeyCredential(self.api_key)
            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=credential,
            )
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

        # Note: Regular document images aren't directly accessible in AnalyzeResult
        # We need to use the figures feature instead

        # Get extracted figures
        if result.figures:
            for i, figure in enumerate(result.figures):
                if not figure.id:
                    continue

                try:
                    # Get figure content
                    response_iter = self._client.get_analyze_result_figure(
                        model_id=result.model_id,
                        result_id=operation_id,
                        figure_id=figure.id,
                    )
                    # Convert iterator to bytes
                    content = b"".join(response_iter)

                    image_id = f"figure-{i}"
                    filename = f"{image_id}.png"

                    images.append(
                        Image(
                            id=image_id,
                            content=content,
                            mime_type="image/png",
                            filename=filename,
                        )
                    )
                except HttpResponseError:
                    logger.warning("Failed to retrieve figure %s", figure.id)
                    continue

        return images

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a document file synchronously using Azure Document Intelligence.

        Args:
            file_path: Path to the file to process
            mime_type: MIME type of the file

        Returns:
            Converted document with extracted text/metadata

        Raises:
            ValueError: If Azure configuration is missing
            HttpResponseError: If Azure API returns an error
        """
        import upath

        path = upath.UPath(file_path)

        # Convert feature strings to enum values
        features = [
            getattr(DocumentAnalysisFeature, feature) for feature in self.features
        ]

        try:
            # Submit document for analysis
            with path.open("rb") as f:
                poller = self._client.begin_analyze_document(
                    model_id=self.model,
                    body=f,
                    features=features,
                    locale=self.languages[0] if self.languages else "en",
                )
            result = poller.result()
            operation_id = poller.details["operation_id"]

            # Extract metadata
            metadata = {}
            if result.documents:
                doc = result.documents[0]  # Get first document
                if doc.fields:
                    metadata = {
                        name: field.get("valueString") or field.get("content", "")
                        for name, field in doc.fields.items()
                    }

            # Convert images
            images = self._convert_azure_images(result, operation_id)

            return Document(
                content=result.content,
                images=images,
                title=path.stem,
                source_path=str(path),
                mime_type=mime_type,
                page_count=len(result.pages) if result.pages else None,
                **metadata,
            )

        except HttpResponseError as e:
            # Add more context to Azure errors
            msg = f"Azure Document Intelligence failed: {e.message}"
            if e.error:
                msg = f"{msg} (Error code: {e.error.code})"
            raise ValueError(msg) from e

    # async def _convert_path_async(self, file_path: StrPath, mime_type: str) -> Document:
    #     """Convert a document file asynchronously - falls back to sync."""
    #     import anyenv

    #     return await anyenv.run_in_thread(self._convert_path_sync, file_path, mime_type)
