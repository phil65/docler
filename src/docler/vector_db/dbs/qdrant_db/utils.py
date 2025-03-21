"""Qdrant vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from qdrant_client.http import models


def get_query(filters: dict[str, Any] | None = None) -> models.Filter | None:
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

    filters = filters or {}
    conditions = []
    for field_name, val in filters.items():
        match = MatchAny(any=val) if isinstance(val, list) else MatchValue(value=val)
        cond = FieldCondition(key=field_name, match=match)
        conditions.append(cond)
    return Filter(must=conditions) if conditions else None
