from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from upath import UPath

from docler.common_types import SupportedLanguage
from docler.converters.base import DocumentConverter


ConverterShorthand = Literal[
    "docling",
    "marker",
    "mistral",
    "llamaparse",
    "datalab",
    "azure",
    "llm",
    "markitdown",
    "upstage",
]


mcp = FastMCP("phil65")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Convert to Markdown",
        readOnlyHint=False,
        idempotentHint=True,
    )
)
async def convert_to_markdown(
    # ctx: Context,
    source_path: str,
    target_folder: str,
    provider: ConverterShorthand = "mistral",
    languages: list[SupportedLanguage] | None = None,
    page_range: str | None = None,
) -> str:
    """Convert a PDF to markdown and images.

    Args:
        source_path: Path to the source document (PDF, powerpoint, images, ...)
        target_folder: Path where to save the resulting markdown and images
        provider: The provider to use for conversion.
        languages: Used languages in the document, if known.
        page_range: 1-based page range, for example "1,4-7,9-10"
    """
    try:
        target = UPath(target_folder).expanduser()
        source = UPath(source_path).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        converters: list[type[DocumentConverter]] = DocumentConverter.get_available_providers()
        converter_cls = next(i for i in converters if provider == i.NAME)
        converter = converter_cls(languages=languages, page_range=page_range)
        # await ctx.info(f"Starting conversion for {source_path} using {provider!r}...")
        document = await converter.convert_file(source)
        # await ctx.info(f"Exporting to {target}...")
        await document.export_to_directory(target)
        files_str = "\n- ".join(str(i) for i in target.iterdir())
    except FileNotFoundError as e:
        return f"Error: {e}. Is the path correct? Current cwd is {UPath.cwd()}."
    else:
        return f"Conversion successful. Files created in {target_folder}:\n{files_str}"


if __name__ == "__main__":
    import logging

    logging.basicConfig()
    mcp.run()
