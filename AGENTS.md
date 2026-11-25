# AGENTS.md

## Overview

Docler is an async Python library (3.13+) that converts documents (PDF, images, office
files) to markdown via one interface over many OCR / document-AI backends (Azure, Mistral,
LlamaParse, Upstage, DataLab, Docling, Marker, MarkItDown, LLMs). It also ships chunkers,
annotators and processors for the converted documents, plus a FastAPI server, an MCP
server, Streamlit UIs and a Typer CLI. The `Document` / `Image` / `TextChunk` data models
come from the external `mkdown` package.

## Architecture

One wheel, five top-level packages (`[tool.uv.build-backend] module-name`):

- `src/docler/`: core library.
  - `provider.py`: `BaseProvider[TConfig]`, the base of everything pluggable.
    `REQUIRED_PACKAGES` (import names, checked via `importlib.util.find_spec`) decides
    whether a provider is available; `get_available_providers()` returns the
    *imported* subclasses, so a provider is only discoverable after its module is imported.
  - `converters/base.py`: `DocumentConverter`. Subclasses implement `_convert_sync` or
    `_convert_async(data: BytesIO, mime_type) -> ConverterResult`; the base class handles
    file/remote I/O (`upathtools`), page ranges and assembles the `mkdown.Document`.
    Public entry points: `convert_file`, `convert_files`, `convert_content`.
    Subclasses self-register in `DocumentConverter.registry` by `NAME`.
  - `converters/<name>_provider/`: one package per backend (`provider.py`, often
    `utils.py` for response parsing). Heavy SDKs are imported lazily inside methods.
  - `converters/registry.py` (`ConverterRegistry`, mime-based selection),
    `converters/aggregated_converter.py` (`AggregatedConverter`, routes per mime type),
    `converters/dir_converter.py` (`DirectoryConverter`, `Conversion` progress state).
  - `chunkers/`: `TextChunker` base (`split` -> `list[TextChunk]`, `chunk` ->
    `ChunkedDocument`), implementations in `ai_chunker/`, `markdown_chunker/`,
    `llamaindex_chunker/`, `token_chunker/`. `chunkers/__init__.py` imports nothing.
  - `annotators/` (`Annotator`; AI document/image annotators), `processors/`
    (`DocumentProcessor`; `LLMProofReader`).
  - `models.py` (`ChunkedDocument`, `PageMetadata`, ...), `common_types.py`,
    `pdf_utils.py` (pypdf: page ranges, metadata, decryption), `utils.py`, `diffs.py`.
  - `__main__.py`: Typer CLI `docler` (`api`, `mcp`, `ocr_ui`, `chunk_ui`).
- `src/docler_config/`: pydantic-settings configs mirroring each provider
  (`ProviderConfig` base in `provider.py`; `converter_configs.py`, `chunker_configs.py`,
  `annotator_configs.py`, `processor_configs.py`). Each config has a `type` discriminator
  and `get_provider()`; providers have `from_config` / `to_config`. `ConverterConfig` etc.
  are discriminated unions.
- `src/docler_api/`: FastAPI app (`main.py` endpoints, `routes.py` implementations).
  Configs are passed as JSON strings in multipart form fields.
- `src/docler_mcp/server.py`: FastMCP server (`convert_to_markdown` tool).
- `src/docler_streamlit/`: Streamlit apps `ocr_app.py`, `chunk_app.py`.

## Where to look

| Task | Location |
|------|----------|
| Add a converter | `src/docler/converters/<name>_provider/`, config in `src/docler_config/converter_configs.py` (add to `ConverterConfig` union), export in `src/docler/__init__.py` |
| Add a chunker | `src/docler/chunkers/<name>/`, `src/docler_config/chunker_configs.py`, import it in `list_chunkers` in `src/docler_api/routes.py` |
| Conversion I/O, page ranges, passwords | `src/docler/converters/base.py`, `src/docler/pdf_utils.py` |
| Page-break / image markup in markdown | `mkdown` package (`create_page_break`, `create_image_reference`) |
| HTTP API | `src/docler_api/main.py`, `src/docler_api/routes.py`, `src/docler_api/README.md` |
| CLI | `src/docler/__main__.py` |
| Docker image | `Dockerfile`, `compose.yml`, `.github/workflows/publish_docker.yml` |
| Tests / snapshots | `tests/`, `tests/__snapshots__/`, recorded API responses in `tests/responses/` |

## Commands

```bash
uv sync --extra all                 # dev env (not --all-extras: see Pitfalls)
uv run pytest                       # unit tests; integration tests are skipped
uv run pytest --integration         # include @pytest.mark.integration (need API keys/services)
uv run pytest --snapshot-update     # update syrupy snapshots
uvx ruff check . && uvx ruff format --check .
uv run ty check                     # must be warning-free; checks src/ only
uv run deptry src                   # dependency hygiene
uv run mknodes build                # docs (duties.py wraps these; duty is not a declared dep: uvx duty ...)
uvx duty lint                       # ruff fix + format + ty (via duties.py)
uv build
```

CI (`.github/workflows/build.yml`) runs `uv sync --extra light --extra server --no-group docs`,
ruff, `ty check` and pytest on 3.13/3.14 (3.14 allowed to fail) x Linux/macOS/Windows.
Docs (`documentation.yml`) sync `--extra all`. Pre-commit (prek) runs ruff, tombi, ty,
pytest and commitizen (conventional commits).

## Conventions

- Google-style docstrings, no types in `Args`; `from __future__ import annotations`
  everywhere; type-only imports under `TYPE_CHECKING` (ruff `TC` rules), except types
  pydantic must resolve at runtime.
- Line length 100, ruff preview formatting. PEP 695 generics (`class X[T: BaseModel]`).
- Tests are plain functions (no classes), async tests via `asyncio_mode = "auto"`.
- API keys come from env vars via `docler.utils.get_api_key` (`MISTRAL_API_KEY`,
  `LLAMAPARSE_API_KEY`, `UPSTAGE_API_KEY`, `DATALAB_API_KEY`,
  `AZURE_DOC_INTELLIGENCE_ENDPOINT`, `AZURE_DOC_INTELLIGENCE_KEY`); LLM-based
  components use pydantic-ai model ids (e.g. `openai:gpt-4o`) through `agentpool`.

## Specs

Remote document-conversion APIs. Local libraries (docling, marker, markitdown, llama-index chunkers) and the LLM-based converters are not listed.

| Spec | URL | Targets | Code | Update check |
|---|---|---|---|---|
| Azure AI Document Intelligence (via `azure-ai-documentintelligence`) | https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout | unpinned (SDK default API version; model `prebuilt-layout` by default) | `src/docler/converters/azure_provider/` | same URL, SDK releases on PyPI |
| Datalab Marker API | https://www.datalab.to/app/reference | API v1 (`https://www.datalab.to/api/v1/marker`) | `src/docler/converters/datalab_provider/` | https://www.datalab.to/openapi.json |
| docling-serve | https://github.com/docling-project/docling-serve/blob/main/docs/usage.md | `/v1alpha/convert/{source,file}` (`docling_remote_provider/provider.py`); current upstream docs use `/v1/convert/...` | `src/docler/converters/docling_remote_provider/` | https://github.com/docling-project/docling-serve/releases |
| LlamaParse (via `llama-parse`) | https://developers.llamaindex.ai/llamaparse/parse/ | unpinned (tracks latest SDK) | `src/docler/converters/llamaparse_provider/` | same URL, `llama-parse` releases on PyPI |
| Mistral OCR (via `mistralai>=3`) | https://docs.mistral.ai/api/endpoint/ocr | unpinned (model `mistral-ocr-latest` by default) | `src/docler/converters/mistral_provider/` | same URL |
| Upstage Document Parse | https://console.upstage.ai/api/document-digitization/document-parsing | API v1 (`https://api.upstage.ai/v1/document-digitization`) | `src/docler/converters/upstage_provider/` | same URL |

## Pitfalls

- Copier-managed project (`.copier-answers.yml`, template `copier-phil65`): workflows,
  `duties.py`, `mkdocs.yml`, pre-commit and most of `pyproject.toml` come from the template.
  Add dependencies below the `# Only add below (Copier)` markers.
- Extras conflict: `marker` (marker-pdf needs `openai<3`) cannot be installed with `all` /
  `light` (agentpool needs `openai>=3.31`); declared in `[tool.uv] conflicts`.
  Never `uv sync --all-extras`.
- Many backends are `python_version < '3.14'` only (docling, llama-parse, marker,
  markitdown, streamlit). Their imports are allowed-unresolved for ty
  (`[tool.ty.analysis]`); keep them lazy (inside functions) so `import docler` stays light.
- `REQUIRED_PACKAGES` holds import names, not distribution names.
- Providers register only when their module is imported; `docler/__init__.py` imports the
  converters, nothing imports the chunkers/annotators/processors implicitly.
- `mkdown.TextChunk` imports `Image` only for type checking, hence the explicit
  `ChunkedDocument.model_rebuild(...)` in `src/docler/models.py`.
- Docs are the mknodes default site: `README.md` is the index page, the rest is
  generated from docstrings; `docs/` is intentionally empty.
