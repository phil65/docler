"""Custom field types with 'field_type' metadata for UI rendering hints."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from schemez import Schema


class ImageClassification(Schema):
    """First-stage classification of an image."""

    image_type: Literal[
        "photo", "diagram", "chart", "graph", "table", "map", "screenshot", "other"
    ]
    """Type of the image."""

    description: str
    """General description of what's in the image."""

    needs_diagram_analysis: bool = False
    """Whether this image should be sent for specialized diagram analysis."""


class DiagramAnalysis(Schema):
    """Second-stage detailed analysis for diagrams."""

    diagram_type: Literal[
        "flowchart",
        "sequence",
        "class",
        "entity_relationship",
        "mindmap",
        "network",
        "architecture",
        "other",
    ]
    """Specific type of diagram."""

    mermaid_code: str
    """A mermaid.js compatible representation of the diagram."""

    key_elements: list[str] = Field(default_factory=list)
    """Important elements/nodes in the diagram."""

    key_insights: list[str] = Field(default_factory=list)
    """Key insights or important aspects of the diagram."""
