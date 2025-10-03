# Docler

[![PyPI License](https://img.shields.io/pypi/l/docler.svg)](https://pypi.org/project/docler/)
[![Package status](https://img.shields.io/pypi/status/docler.svg)](https://pypi.org/project/docler/)
[![Monthly downloads](https://img.shields.io/pypi/dm/docler.svg)](https://pypi.org/project/docler/)
[![Distribution format](https://img.shields.io/pypi/format/docler.svg)](https://pypi.org/project/docler/)
[![Wheel availability](https://img.shields.io/pypi/wheel/docler.svg)](https://pypi.org/project/docler/)
[![Python version](https://img.shields.io/pypi/pyversions/docler.svg)](https://pypi.org/project/docler/)
[![Implementation](https://img.shields.io/pypi/implementation/docler.svg)](https://pypi.org/project/docler/)
[![Releases](https://img.shields.io/github/downloads/phil65/docler/total.svg)](https://github.com/phil65/docler/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/docler)](https://github.com/phil65/docler/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/docler)](https://github.com/phil65/docler/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/docler)](https://github.com/phil65/docler/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/docler)](https://github.com/phil65/docler/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/docler)](https://github.com/phil65/docler/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/docler)](https://github.com/phil65/docler/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/docler)](https://github.com/phil65/docler/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/docler)](https://github.com/phil65/docler)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/docler)](https://github.com/phil65/docler/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/docler)](https://github.com/phil65/docler/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/docler)](https://github.com/phil65/docler)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/docler)](https://github.com/phil65/docler)
[![Package status](https://codecov.io/gh/phil65/docler/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/docler/)
[![PyUp](https://pyup.io/repos/github/phil65/docler/shield.svg)](https://pyup.io/repos/github/phil65/docler/)

[Read the documentation!](https://phil65.github.io/docler/)

A unified Python library for document conversion and OCR that provides a consistent interface to multiple document processing providers. Extract text, images, and metadata from PDFs, images, and office documents using state-of-the-art OCR and document AI services.

## Features

- **Unified Interface**: Single API for multiple document processing providers
- **Multiple Providers**: Support for 10+ OCR and document AI services
- **Rich Output**: Extract text, images, tables, and metadata
- **Async Support**: Built-in async/await support
- **Flexible Configuration**: Provider-specific settings and preferences
- **Page Range Support**: Process specific pages from documents
- **Multi-language OCR**: Support for 100+ languages across providers
- **Structured Output**: Standardized markdown with embedded metadata

## Quick Start

```python
import asyncio
from docler import MistralConverter

async def main():
    # Use the aggregated converter for automatic provider selection
    converter = MistralConverter()

    # Convert a document
    result = await converter.convert_file("document.pdf")

    print(f"Title: {result.title}")
    print(f"Content: {result.content[:500]}...")
    print(f"Images: {len(result.images)} extracted")
    print(f"Pages: {result.page_count}")

asyncio.run(main())
```

## Available OCR Converters

### Cloud API Providers

#### Azure Document Intelligence

```python
from docler import AzureConverter

converter = AzureConverter(
    endpoint="your-endpoint",
    api_key="your-key",
    model="prebuilt-layout"
)
```

#### Mistral OCR

```python
from docler import MistralConverter

converter = MistralConverter(
    api_key="your-key",
    languages=["en", "fr", "de"]
)
```

#### LlamaParse

```python
from docler import LlamaParseConverter

converter = LlamaParseConverter(
    api_key="your-key",
    adaptive_long_table=True
)
```

#### Upstage Document AI

```python
from docler import UpstageConverter

converter = UpstageConverter(
    api_key="your-key",
    chart_recognition=True
)
```

#### DataLab

```python
from docler import DataLabConverter

converter = DataLabConverter(
    api_key="your-key",
    use_llm=False  # Enable for higher accuracy
)
```

### Local/Self-Hosted Providers

#### Marker

```python
from docler import MarkerConverter

converter = MarkerConverter(
    dpi=192,
    use_llm=True,  # Requires local LLM setup
    llm_provider="ollama"
)
```

#### Docling

```python
from docler import DoclingConverter

converter = DoclingConverter(
    ocr_engine="easy_ocr",
    image_scale=2.0
)
```

#### Docling Remote

```python
from docler import DoclingRemoteConverter

converter = DoclingRemoteConverter(
    endpoint="http://localhost:5001",
    pdf_backend="dlparse_v4"
)
```

#### MarkItDown (Microsoft)

```python
from docler import MarkItDownConverter

converter = MarkItDownConverter()
```

### LLM-Based Providers

#### LLM Converter

```python
from docler import LLMConverter

converter = LLMConverter(
    model="gpt-4o",  # or claude-3-5-sonnet, etc.
    system_prompt="Extract text preserving formatting..."
)
```

## Provider Comparison

| Provider | Cost/Page | Local | API Required | Best For |
|----------|-----------|-------|--------------|----------|
| **Azure** | $0.0096 | ❌ | ✅ | Enterprise forms, invoices |
| **Mistral** | Variable | ❌ | ✅ | High-quality text extraction |
| **LlamaParse** | $0.0045 | ❌ | ✅ | Complex layouts, academic papers |
| **Upstage** | $0.01 | ❌ | ✅ | Charts, presentations |
| **DataLab** | $0.0015 | ❌ | ✅ | Cost-effective processing |
| **Marker** | Free | ✅ | ❌ | Privacy-sensitive documents |
| **Docling** | Free | ✅ | ❌ | Open-source processing |
| **MarkItDown** | Free | ✅ | ❌ | Office documents |
| **LLM** | Variable | ❌ | ✅ | Latest AI capabilities |

## Advanced Usage

### Directory Processing

Process entire directories with progress tracking:

```python
from docler import DirectoryConverter, MarkerConverter

base_converter = MarkerConverter()
dir_converter = DirectoryConverter(base_converter, chunk_size=10)

# Convert all supported files
results = await dir_converter.convert("./documents/")

# Or with progress tracking
async for state in dir_converter.convert_with_progress("./documents/"):
    print(f"Progress: {state.processed_files}/{state.total_files}")
    print(f"Current: {state.current_file}")
    if state.errors:
        print(f"Errors: {len(state.errors)}")
```

### Page Range Processing

Extract specific pages from documents:

```python
# Extract pages 1-5 and 10-15
converter = MistralConverter(page_range="1-5,10-15")
result = await converter.convert_file("large_document.pdf")
```

### Batch Processing

Process multiple files efficiently:

```python
files = ["doc1.pdf", "doc2.png", "doc3.docx"]
results = await converter.convert_files(files)

for file, result in zip(files, results):
    print(f"{file}: {len(result.content)} characters extracted")
```

## Output Format

All converters return a standardized `Document` object with:

```python
class Document:
    content: str           # Extracted text in markdown format
    images: list[Image]    # Extracted images with metadata
    title: str            # Document title
    source_path: str      # Original file path
    mime_type: str        # File MIME type
    metadata: dict        # Provider-specific metadata
    page_count: int       # Number of pages processed
```

The markdown content includes standardized metadata for page breaks and structure:

```markdown
<!-- docler:page_break {"next_page":1} -->
# Document Title

Content from page 1...

<!-- docler:page_break {"next_page":2} -->
More content from page 2...
```

## Installation

```bash
# Basic installation
pip install docler

# With specific provider dependencies
pip install docler[azure]      # Azure Document Intelligence
pip install docler[mistral]    # Mistral OCR
pip install docler[marker]     # Marker PDF processing
pip install docler[all]        # All providers
```

## Environment Variables

Configure API keys via environment variables:

```bash
export AZURE_DOC_INTELLIGENCE_ENDPOINT="your-endpoint"
export AZURE_DOC_INTELLIGENCE_KEY="your-key"
export MISTRAL_API_KEY="your-key"
export LLAMAPARSE_API_KEY="your-key"
export UPSTAGE_API_KEY="your-key"
export DATALAB_API_KEY="your-key"
```

## Contributing

We welcome contributions! See our [contributing guidelines](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Documentation**: https://phil65.github.io/docler/
- **PyPI**: https://pypi.org/project/docler/
- **GitHub**: https://github.com/phil65/docler/
- **Issues**: https://github.com/phil65/docler/issues
- **Discussions**: https://github.com/phil65/docler/discussions

---

**Coming Soon**: FastAPI demo with bring-your-own-keys on https://contexter.net
