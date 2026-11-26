"""DataLab API client: submit, then poll the check URL (against a local server)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from docler.converters.datalab_provider import utils


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.connections = 0
        self.polls = 0
        self.api_keys: list[str | None] = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def setup(self) -> None:
        super().setup()
        self.server.connections += 1

    def log_message(self, *args: object) -> None:
        pass

    def _json(self, data: object) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.api_keys.append(self.headers.get("X-Api-Key"))
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._json({"success": True, "request_check_url": f"{base}/check/1"})

    def do_GET(self) -> None:
        self.server.api_keys.append(self.headers.get("X-Api-Key"))
        self.server.polls += 1
        if self.server.polls < 3:  # noqa: PLR2004
            self._json({"status": "processing"})
        else:
            self._json({"status": "complete", "success": True, "markdown": "# Hi"})


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    srv = _Server()
    threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
    monkeypatch.setattr(utils, "API_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(utils, "POLL_INTERVAL", 0)
    yield srv
    srv.shutdown()
    srv.server_close()


async def test_get_response_submits_and_polls_over_one_connection(server: _Server):
    result = await utils.get_response({"output_format": "markdown"}, {"file": b"%PDF"}, "key")
    assert result["markdown"] == "# Hi"
    assert server.polls == 3  # noqa: PLR2004
    assert server.api_keys == ["key"] * 4
    assert server.connections == 1
