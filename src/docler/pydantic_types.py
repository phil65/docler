"""Custom field types with 'field_type' metadata for UI rendering hints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
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


# Helper function to extract field type metadata
def get_field_type(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    """Extract field_type metadata from a model field."""
    field_info = model.model_fields[field_name]
    metadata = {}
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        metadata.update(field_info.json_schema_extra)

    return metadata


def render_field(model: type[BaseModel], field_name: str) -> str:
    """Example function demonstrating how to use field type metadata for UI rendering."""
    metadata = get_field_type(model, field_name)
    field_type = metadata.get("field_type", "text")
    if field_type == "model_identifier":
        provider = metadata.get("provider")
        if provider:
            return f"Model selector dropdown for {provider} provider"
        return "Generic model identifier selector"

    return "Default text input"
