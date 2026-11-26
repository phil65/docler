"""ASGI middleware: bearer-token auth and the upload size limit.

Both run before the request body is read, so an unauthenticated or oversized upload is
refused without being parsed or spooled.
"""

from __future__ import annotations

import contextlib
import hmac
from typing import TYPE_CHECKING, Any

import anyenv

from docler_api.settings import TOKEN_ENV, is_loopback


if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from docler_api.settings import ApiSettings


OPEN_PATHS = frozenset({"/health", "/ready"})


class _BodyTooLargeError(Exception):
    pass


async def _send_json(send: Send, status: int, body: dict[str, Any], **headers: str) -> None:
    raw = anyenv.dump_json(body).encode()
    header_list = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(raw))]
    header_list += [(k.encode(), v.encode()) for k, v in headers.items()]
    await send({"type": "http.response.start", "status": status, "headers": header_list})
    await send({"type": "http.response.body", "body": raw})


class ApiGuard:
    """Checks the bearer token and limits the request body size."""

    def __init__(self, app: ASGIApp, settings: ApiSettings) -> None:
        self.app = app
        self.settings = settings

    def _auth_error(self, scope: Scope) -> tuple[int, str] | None:
        token = self.settings.token
        if not token:
            # Without a token only local connections are served (a server started with
            # uvicorn directly on a public address, bypassing `docler api`'s check).
            server = scope.get("server")
            if is_loopback(server[0] if server else None):
                return None
            return 403, f"Set {TOKEN_ENV} to serve the Docler API on a non-loopback address"
        headers = dict(scope["headers"])
        scheme, _, sent = headers.get(b"authorization", b"").decode("latin-1").partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(sent.encode(), token.encode()):
            return None
        return 401, "Missing or invalid bearer token"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        if error := self._auth_error(scope):
            status, detail = error
            extra = {"www-authenticate": "Bearer"} if status == 401 else {}  # noqa: PLR2004
            await _send_json(send, status, {"detail": detail}, **extra)
            return

        limit = self.settings.max_upload_bytes
        too_large = {"detail": f"Request body larger than {limit} bytes"}
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > limit:
            await _send_json(send, 413, too_large)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if exceeded:  # the app's answer to the aborted body; ours is the 413
                return
            started = True
            await send(message)

        with contextlib.suppress(_BodyTooLargeError):
            await self.app(scope, limited_receive, guarded_send)
        if exceeded and not started:
            await _send_json(send, 413, too_large)
