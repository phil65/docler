"""Custom field types with 'field_type' metadata for UI rendering hints."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ModelIdentifier = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "model_identifier"},
        pattern=r"^[a-zA-Z0-9\-]+(/[a-zA-Z0-9\-]+)*(:[\w\-\.]+)?$",
        examples=["openai:gpt-o1-mini", "anthropic/claude-3-opus"],
        description="Identifier for an AI model, optionally including provider.",
    ),
]

ModelTemperature = Annotated[
    float,
    Field(
        json_schema_extra={"field_type": "temperature", "step": 0.1},
        ge=0.0,
        le=2.0,
        description=(
            "Controls randomness in model responses.\n"
            "Lower values are more deterministic, higher values more creative"
        ),
        examples=[0.0, 0.7, 1.0],
    ),
]

MimeType = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "mime_type"},
        pattern=r"^[a-z]+/[a-z0-9\-+.]+$",
        examples=["text/plain", "application/pdf", "image/jpeg", "application/json"],
        description="Standard MIME type identifying file formats and content types",
    ),
]


class ImageClassification(BaseModel):
    """First-stage classification of an image."""

    image_type: Literal[
        "photo", "diagram", "chart", "graph", "table", "map", "screenshot", "other"
    ]
    """Type of the image."""

    description: str
    """General description of what's in the image."""

    needs_diagram_analysis: bool = False
    """Whether this image should be sent for specialized diagram analysis."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class DiagramAnalysis(BaseModel):
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

    model_config = ConfigDict(use_attribute_docstrings=True)


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


if __name__ == "__main__":

    class AIConfig(BaseModel):
        """AI Configuration with semantically typed fields."""

        model: ModelIdentifier = "gpt-4"

    config = AIConfig(model="gpt-4")

    print(type(config.model))
