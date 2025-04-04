"""Qdrant vector store backend implementation."""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models

from docler.models import SearchResult


def get_query(filters: dict[str, Any] | None = None) -> models.Filter | None:
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

    filters = filters or {}
    conditions = []
    for field_name, val in filters.items():
        match = MatchAny(any=val) if isinstance(val, list) else MatchValue(value=val)
        cond = FieldCondition(key=field_name, match=match)
        conditions.append(cond)
    return Filter(must=conditions) if conditions else None


def to_pointstructs(vectors, ids, metadata):
    points = []
    for i, vector in enumerate(vectors):
        vector_ls = vector.astype(float).tolist()
        struct = models.PointStruct(id=ids[i], vector=vector_ls, payload=metadata[i])
        points.append(struct)
    return points


def to_search_result(result: models.ScoredPoint) -> SearchResult:
    data = result.payload or {}
    text = data.pop("text", None) if data else None
    txt = str(text) if text is not None else None
    id_ = str(result.id)
    return SearchResult(chunk_id=id_, score=result.score, metadata=data, text=txt)
