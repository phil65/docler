"""LlamaParse response processing (llama_index is faked: it needs Python < 3.14)."""

import sys
import types
from typing import Any

import httpx2
import pytest

from docler.converters.llamaparse_provider import utils


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Client]:
    constants = types.ModuleType("llama_index.core.constants")
    constants.DEFAULT_BASE_URL = "https://api.cloud.llamaindex.ai"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core.constants", constants)

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer key"
        return httpx2.Response(200, content=request.url.path.encode())

    created: list[httpx2.Client] = []
    real_client = httpx2.Client

    def client(**kwargs: Any) -> httpx2.Client:
        created.append(real_client(transport=httpx2.MockTransport(handle), **kwargs))
        return created[-1]

    monkeypatch.setattr(utils.httpx2, "Client", client)
    return created


def test_images_are_fetched_over_one_client(clients: list[httpx2.Client]):
    images = [{"name": f"img_{i}.jpg"} for i in range(3)]
    result = [{"job_id": "j1", "pages": [{"md": "# One", "images": images}, {"md": "two"}]}]
    parts, found = utils.process_response(result, "key")
    assert len(clients) == 1
    assert [i.filename for i in found] == ["img-0.jpg", "img-1.jpg", "img-2.jpg"]
    assert found[0].mime_type == "image/jpg"
    assert "# One" in parts
    assert "two" in parts
