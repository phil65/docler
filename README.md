# Docler

[![PyPI License](https://img.shields.io/pypi/l/docler.svg)](https://pypi.org/project/docler/)
[![Package status](https://img.shields.io/pypi/status/docler.svg)](https://pypi.org/project/docler/)
[![Daily downloads](https://img.shields.io/pypi/dd/docler.svg)](https://pypi.org/project/docler/)
[![Weekly downloads](https://img.shields.io/pypi/dw/docler.svg)](https://pypi.org/project/docler/)
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
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/docler)](https://github.com/phil65/docler)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/docler)](https://github.com/phil65/docler)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/docler)](https://github.com/phil65/docler)
[![Package status](https://codecov.io/gh/phil65/docler/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/docler/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/docler/shield.svg)](https://pyup.io/repos/github/phil65/docler/)

[Read the documentation!](https://phil65.github.io/docler/)


## Markdown Conventions for OCR Output

This project utilizes Markdown as the primary, self-contained format for storing OCR results and associated metadata. The goal is to have a single, versionable, human-readable file representing a processed document, simplifying pipeline management and data provenance.

We employ a hybrid approach, using different mechanisms for different types of metadata:

### 1. Metadata Comments (for Non-Visual Markers)

For metadata that should *not* affect the visual rendering of the Markdown (like page boundaries or page-level information), we use specially formatted HTML/XML comments.

**Format:**

```
<!-- prefix:data_type {compact_json_payload} -->
```

*   **`prefix`**: A namespace identifier to prevent clashes. Defaults to `docler`.
*   **`data_type`**: A string indicating the kind of metadata (e.g., `page_break`, `page_meta`).
*   **`{compact_json_payload}`**: A standard JSON object containing the metadata key-value pairs, serialized compactly (no unnecessary whitespace, keys sorted).

**Defined Types:**

*   **`page_break`**: Marks the transition *to* the specified page number. Placed immediately *before* the content of the new page.
    *   Example Payload: `{"next_page": 2}`
    *   Example Comment: `<!-- docler:page_break {"next_page":2} -->`
*   **`page_meta`**: Contains metadata specific to a page (e.g., dimensions, confidence). Often placed near the beginning of the page's content or alongside the `page_break` comment.
    *   Example Payload: `{"page_num": 1, "width": 612, "height": 792, "confidence": 0.98}`
    *   Example Comment: `<!-- docler:page_meta {"confidence":0.98,"height":792,"page_num":1,"width":612} -->`

### 2. HTML Figures (for Images and Diagrams)

For visual elements like images or diagrams, especially when they require richer metadata (like source code or bounding boxes), we use standard HTML structures within the Markdown. This allows direct association of metadata and handles complex data like code snippets gracefully.

**Structure:**

We typically use an HTML `<figure>` element:

```html
<figure data-docler-type="diagram" data-diagram-id="sysarch-01">
  <img src="images/system_architecture.png"
       alt="System Architecture Diagram"
       data-page-num="5"
       style="max-width: 100%; height: auto;"
       >
  <figcaption>Figure 2: High-level system data flow.</figcaption>
  <script type="text/docler-mermaid">
    graph LR
        A[Data Ingest] --> B(Processing Queue);
        B --> C{Main Processor};
        D --> F(API Endpoint);
  </script>
</figure>
```

*   **`<figure>`**: The container element.
    *   `data-docler-type`: Indicates the type of figure (e.g., `image`, `diagram`).
    *   Other `data-*` attributes can be added for figure-level metadata.
*   **`<img>`**: The visual representation.
    *   `src`, `alt`: Standard attributes.
    *   `data-*`: Used for image-specific metadata like `data-page-num`
    *   `style`: Optional for basic presentation.
*   **`<figcaption>`**: Optional standard HTML caption.
*   **`<script type="text/docler-...">`**: Used to embed source code or other complex textual data.
    *   The `type` attribute is custom (e.g., `text/docler-mermaid`, `text/docler-latex`) so browsers ignore it.
    *   The raw code/text is placed inside, preserving formatting.

### Rationale

*   **Comments** are used for page breaks and metadata because they are guaranteed *not* to interfere with Markdown rendering, ensuring purely structural information remains invisible.
*   **HTML Figures** are used for images/diagrams because HTML provides standard ways (`data-*`, nested elements like `<script>`) to directly associate rich, potentially complex or multi-line metadata (like source code) with the visual element itself.

### Utilities

Helper functions for creating and parsing these metadata comments and structures are available in `mkdown`.

### Standardized Metadata Types

The library provides standardized metadata types for common use cases:

1. **Page Breaks**: Use `create_page_break()` function to create page transitions:
   ```python
   from mkdown import create_page_break

   # Create a page break marker for page 2
   page_break = create_page_break(next_page=2)
   # <!-- docler:page_break {"next_page":2} -->
   ```

2. **Chunk Boundaries**: Use `create_chunk_boundary()` function to mark semantic chunks in a document:
   ```python
   from mkdown import create_chunk_boundary

   # Create a chunk boundary marker with metadata
   chunk_marker = create_chunk_boundary(
       chunk_id=1,
       start_line=10,
       end_line=25,
       keywords=["introduction", "overview"],
       token_count=350,
   )
   # <!-- docler:chunk_boundary {"chunk_id":1,"end_line":25,"keywords":["introduction","overview"],"start_line":10,"token_count":350} -->
   ```
