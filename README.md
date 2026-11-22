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

A unified, async Python library for document conversion and OCR. Docler wraps many OCR and
document-AI services behind one interface and turns PDFs, images and office documents into
markdown with extracted images and metadata.

## Features

- **Unified interface**: one `DocumentConverter` API over 10 cloud and local backends
- **Standardized output**: markdown with page-break markers, extracted images and metadata
- **Async first**: convert single files, batches or whole directories (local or remote via fsspec)
- **Page ranges and encrypted PDFs**: `page_range="1-5,10"`, `pdf_password=...`
- **Chunking**: markdown-, token-, LlamaIndex- and LLM-based chunkers
- **Serializable configs**: every provider has a pydantic config model (`docler_config`)
- **Ready-made frontends**: FastAPI server, MCP server, Streamlit UIs, `docler` CLI

## Installation

```bash
pip install docler                # core (no backend SDKs)
pip install "docler[mistralai]"   # Mistral OCR
pip install "docler[light]"       # cloud backends (Mistral, LlamaParse, Azure, ...) + LLM features
pip install "docler[all]"         # everything except Marker
pip install "docler[marker]"      # Marker, in a separate environment (see below)
```

Further extras: `azure`, `llama-parse`, `docling`, `markitdown`, `openai`, `diffs`,
`server` (FastAPI), `mcp`, `streamlit`, `token-chunking`, `olmocr`. Local backends such as
Docling, Marker and MarkItDown currently require Python < 3.14.

Marker cannot be installed together with `docler[all]`/`docler[light]`: marker-pdf
requires `openai<3`, while the AI features (agentpool) require `openai>=3.31`.

## Quick Start

```python
import asyncio

from docler import MistralConverter


async def main():
    converter = MistralConverter()  # reads MISTRAL_API_KEY

    doc = await converter.convert_file("document.pdf")

    print(f"Title: {doc.title}")
    print(f"Content: {doc.content[:500]}...")
    print(f"Images: {len(doc.images)} extracted")


asyncio.run(main())
```

Without an API key, `MarkItDownConverter` (`docler[markitdown]`) works fully locally.

## Available OCR Converters

### Cloud API Providers

#### Azure Document Intelligence

```python
from docler import AzureConverter

converter = AzureConverter(endpoint="your-endpoint", api_key="your-key", model="prebuilt-layout")
```

#### Mistral OCR

```python
from docler import MistralConverter

converter = MistralConverter(api_key="your-key", languages=["en", "fr", "de"])
```

#### LlamaParse

```python
from docler import LlamaParseConverter

converter = LlamaParseConverter(api_key="your-key", adaptive_long_table=True)
```

#### Upstage Document AI

```python
from docler import UpstageConverter

converter = UpstageConverter(api_key="your-key", chart_recognition=True)
```

#### DataLab

```python
from docler import DataLabConverter

converter = DataLabConverter(
    api_key="your-key",
    use_llm=False,  # Enable for higher accuracy
)
```

### Local/Self-Hosted Providers

#### Marker

```python
from docler import MarkerConverter

converter = MarkerConverter(
    dpi=192,
    use_llm=True,  # Requires local LLM setup
    llm_provider="ollama",
)
```

#### Docling

```python
from docler import DoclingConverter

converter = DoclingConverter(ocr_engine="rapid_ocr", image_scale=2.0)
```

#### Docling Remote

```python
from docler.converters.docling_remote_provider import DoclingRemoteConverter

converter = DoclingRemoteConverter(endpoint="http://localhost:5001", pdf_backend="dlparse_v4")
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
    model="openai:gpt-4o",  # any pydantic-ai model identifier
    system_prompt="Extract text preserving formatting...",
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

`AggregatedConverter` picks a suitable registered converter per MIME type:

```python
from docler import AggregatedConverter

converter = AggregatedConverter()
converter.set_converter_preference("application/pdf", "mistral")
doc = await converter.convert_file("document.pdf")
```

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

### Chunking

```python
from docler.chunkers.markdown_chunker import MarkdownChunker

chunker = MarkdownChunker(max_chunk_size=1500)
chunked = await chunker.chunk(doc)  # ChunkedDocument

for chunk in chunked.chunks:
    print(chunk.chunk_index, chunk.page_number, len(chunk.content), len(chunk.images))
```

Other chunkers: `TokenAwareChunker` (`docler.chunkers.token_chunker`, extra
`token-chunking`), `LlamaIndexChunker` (`docler.chunkers.llamaindex_chunker`) and
`AIChunker` (`docler.chunkers.ai_chunker`, LLM-based).

### Configuration objects

Every provider has a serializable pydantic config, handy for config files and the HTTP API:

```python
from docler_config.converter_configs import MistralConfig

converter = MistralConfig(languages=["en"]).get_provider()
```

## Server, MCP and CLI

```bash
docler api --host 0.0.0.0 --port 8000   # FastAPI server (extra: server)
docler mcp --transport stdio            # MCP server with a convert_to_markdown tool (extra: mcp)
docler ocr_ui                           # Streamlit OCR UI (extra: streamlit)
docler chunk_ui                         # Streamlit chunking UI
```

See [`src/docler_api/README.md`](https://github.com/phil65/docler/blob/main/src/docler_api/README.md) for the API endpoints. A
`Dockerfile` and `compose.yml` are included for running the API server.

## Output Format

All converters return a standardized `Document` object with:

```python
class Document:  # from the mkdown package
    content: str  # Extracted text in markdown format
    images: list[Image]  # Extracted images (id, content, mime_type, filename, ...)
    title: str | None  # Document title
    author: str | None
    created: datetime | None
    modified: datetime | None
    source_path: str | None  # Original file path
    mime_type: str | None  # File MIME type
    metadata: dict[str, Any]  # Provider-specific metadata
```

The markdown content includes standardized metadata for page breaks and structure:

```markdown
<!-- docler:page_break {"next_page":1} -->
# Document Title

Content from page 1...

<!-- docler:page_break {"next_page":2} -->
More content from page 2...
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

LLM-based components (`LLMConverter`, `AIChunker`, annotators) use the usual provider
variables of pydantic-ai, e.g. `OPENAI_API_KEY` or `GEMINI_API_KEY`.

## Contributing

Contributions are welcome. See [AGENTS.md](https://github.com/phil65/docler/blob/main/AGENTS.md) for the project layout and dev commands.

## License

MIT License - see [LICENSE](https://github.com/phil65/docler/blob/main/LICENSE) for details.

## Links

- **Documentation**: https://phil65.github.io/docler/
- **PyPI**: https://pypi.org/project/docler/
- **GitHub**: https://github.com/phil65/docler/
- **Issues**: https://github.com/phil65/docler/issues
- **Discussions**: https://github.com/phil65/docler/discussions

---

**Coming Soon**: FastAPI demo with bring-your-own-keys on https://contexter.net
