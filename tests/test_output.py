import tempfile
from typing import TYPE_CHECKING

import pytest
import upath

from docler.converters.datalab_provider import DataLabConverter


if TYPE_CHECKING:
    from docler.models import Document


# Add other providers as needed

PROVIDERS = [
    DataLabConverter,
    # Add other provider classes here
]


@pytest.mark.integration
@pytest.mark.parametrize("provider_cls", PROVIDERS)
@pytest.mark.asyncio
async def test_export_to_directory_snapshot(provider_cls, snapshot):
    sample_path = "src/docler/resources/pdf_sample.pdf"
    provider = provider_cls()
    doc: Document = await provider.convert_file(sample_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        await doc.export_to_directory(tmpdir)
        base = upath.UPath(tmpdir)
        # Find markdown file
        md_files = list(base.glob("*.md"))
        assert md_files, "No markdown file exported"
        md_content = md_files[0].read_text(encoding="utf-8")
        # List all files (relative paths)
        file_list = sorted(
            str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()
        )
        # Snapshot tuple: (markdown content, file list)
        assert (md_content, file_list) == snapshot
