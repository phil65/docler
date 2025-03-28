"""Custom field types with 'field_type' metadata for UI rendering hints."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field


ModelIdentifier = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "model_identifier"},
        pattern=r"^[a-zA-Z0-9\-]+(/[a-zA-Z0-9\-]+)*(:[\w\-\.]+)?$",
    ),
]

Temperature = Annotated[
    float,
    Field(json_schema_extra={"field_type": "parameter", "step": 0.1}, ge=0.0, le=2.0),
]


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
