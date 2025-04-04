from __future__ import annotations

from typing import TYPE_CHECKING

from docler.models import SearchResult


if TYPE_CHECKING:
    from chromadb import QueryResult


def to_search_results(results: QueryResult) -> list[SearchResult]:
    distances = results.get("distances")
    metadatas = results.get("metadatas")

    assert distances is not None
    assert distances[0]
    assert metadatas is not None
    assert metadatas[0]
    search_results: list[SearchResult] = []

    for i, doc_id in enumerate(results["ids"][0]):
        metadata = dict(metadatas[0][i])
        text = metadata.pop("text", None)
        if text is not None:
            text = str(text)
        score = 1.0 - float(distances[0][i])
        id_ = str(doc_id)
        result = SearchResult(chunk_id=id_, score=score, metadata=metadata, text=text)
        search_results.append(result)
    return search_results
