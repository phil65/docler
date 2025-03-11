"""OCR functionality using OLM's OCR model."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from docler.base import DocumentConverter
from docler.models import Document, Image as DoclerImage


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


logger = logging.getLogger(__name__)
PdfEngine = Literal["pdftotext", "pdfium", "pypdf", "topcoherency", "pdfreport"]


class OlmConverter(DocumentConverter):
    """Document converter using OLM's OCR model."""

    NAME = "olm"
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        model_name: str = "allenai/olmOCR-7B-0225-preview",
        device: str | None = None,
        engine: PdfEngine = "pdfreport",
    ) -> None:
        """Initialize the OLM converter.

        Args:
            languages: List of supported languages.
            model_name: Name of the OLM model to use.
            device: Device to run model on ("cuda", "cpu", etc).
                If None, will use CUDA if available.
            engine: PDF engine to use.
        """
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        super().__init__(languages=languages)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Loading OLM model on %s...", self.device)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.engine = engine
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _convert_page(
        self,
        pdf_path: str,
        page_num: int,
    ) -> tuple[str, list[DoclerImage]]:
        """Convert a single PDF page."""
        from olmocr.data.renderpdf import render_pdf_to_base64png
        from olmocr.prompts import build_finetuning_prompt
        from olmocr.prompts.anchor import get_anchor_text
        from PIL import Image

        # Convert page to PNG
        image_base64 = render_pdf_to_base64png(
            pdf_path,
            page_num,
            target_longest_image_dim=1024,
        )

        # Build prompt with anchor text
        anchor_text = get_anchor_text(
            pdf_path,
            page_num,
            pdf_engine=self.engine,  # pyright: ignore
            target_length=4000,
        )
        prompt = build_finetuning_prompt(anchor_text)

        # Prepare model inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate output
        output = self.model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=8192,
            num_return_sequences=1,
            do_sample=True,
        )

        # Decode and parse output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = self.processor.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]

        try:
            text_output = json.loads(text_output)
            content = text_output["natural_text"]
        except Exception:  # noqa: BLE001
            try:
                content = text_output.split("natural_text")[1].strip()
            except Exception:  # noqa: BLE001
                content = ""

        # Create image object
        image = DoclerImage(
            id=f"page_{page_num}",
            content=image_base64,
            mime_type="image/png",
            filename=f"page_{page_num}.png",
        )

        return content, [image]

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a PDF file using OLM OCR."""
        from pypdf import PdfReader
        import upath

        pdf_path = str(upath.UPath(file_path))

        # Get page count
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)

        # Process all pages
        contents: list[str] = []
        images: list[DoclerImage] = []

        for page_num in range(1, page_count + 1):
            content, page_images = self._convert_page(pdf_path, page_num)
            contents.append(content)
            images.extend(page_images)

        return Document(
            content="\n\n".join(contents),
            images=images,
            title=upath.UPath(file_path).stem,
            source_path=str(file_path),
            mime_type=mime_type,
            page_count=page_count,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = OlmConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
