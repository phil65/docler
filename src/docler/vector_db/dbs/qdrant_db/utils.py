"""Qdrant vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from qdrant_client.http import models


def get_query(filters: dict[str, Any] | None = None) -> models.Filter | None:
    from qdrant_client.http import models

    filters = filters or {}
    conditions = []
    for field_name, value in filters.items():
        if isinstance(value, list):
            match = models.MatchAny(any=value)
        else:
            match = models.MatchValue(value=value)
        cond = models.FieldCondition(key=field_name, match=match)
        conditions.append(cond)
    return models.Filter(must=conditions) if conditions else None
