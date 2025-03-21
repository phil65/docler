"""OpenAI Vector Store implementation."""

from __future__ import annotations

from typing import Any


def convert_filters(filters: dict[str, Any]) -> Any:  # type: ignore
    """Convert standard filters to OpenAI's filter format.

    Args:
        filters: Dictionary of filters

    Returns:
        OpenAI-compatible filter object
    """
    filter_conditions = []
    for key, value in filters.items():
        if isinstance(value, list):
            # Handle list of values (OR condition)
            or_conditions = [
                {"key": key, "type": "eq", "value": item}
                for item in value
                if isinstance(item, str | int | float | bool)
            ]

            if or_conditions:
                filter_conditions.append({"type": "or", "filters": or_conditions})
        elif isinstance(value, str | int | float | bool):
            # Handle single value (equality)
            filter_conditions.append({"key": key, "type": "eq", "value": value})  # type: ignore
    if len(filter_conditions) > 1:
        return {"type": "and", "filters": filter_conditions}
    if filter_conditions:
        return filter_conditions[0]
    return {}
