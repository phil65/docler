"""Process runner with configurable wait conditions."""

from __future__ import annotations

import asyncio
import http.client
import re
import socket
from typing import TYPE_CHECKING, Self
import urllib.parse


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

RegexPattern = str


class ProcessRunner:
    def __init__(
        self,
        command: list[str] | str,
        wait_http: list[str] | None = None,  # ["http://localhost:8000/health"]
        wait_tcp: list[tuple[str, int]] | None = None,  # [("localhost", 6379)]
        wait_predicates: list[Callable[[], bool] | Callable[[], Awaitable[bool]]]
        | None = None,
        wait_output: list[RegexPattern] | None = None,
        wait_stderr: list[RegexPattern] | None = None,
        wait_timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        self.command = command if isinstance(command, list) else command.split()
        self.wait_http = wait_http or []
        self.wait_tcp = wait_tcp or []
        self.wait_predicates = wait_predicates or []
        self.wait_output = [re.compile(p) for p in (wait_output or [])]
        self.wait_stderr = [re.compile(p) for p in (wait_stderr or [])]
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.process: asyncio.subprocess.Process | None = None
        self._stdout_patterns_found = dict.fromkeys(self.wait_output, False)
        self._stderr_patterns_found = dict.fromkeys(self.wait_stderr, False)

    async def _check_http(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.netloc)
        try:
            conn.request("GET", parsed.path or "/")
            response = conn.getresponse()
        except (ConnectionRefusedError, socket.gaierror):
            return False
        else:
            return 200 <= response.status < 400  # noqa: PLR2004
        finally:
            conn.close()

    async def _check_tcp(self, host: str, port: int) -> bool:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
        except (ConnectionRefusedError, socket.gaierror):
            return False
        else:
            return True

    async def _check_predicate(
        self, pred: Callable[[], bool] | Callable[[], Awaitable[bool]]
    ) -> bool:
        if asyncio.iscoroutinefunction(pred):
            return await pred()
        return await asyncio.to_thread(pred)  # type: ignore

    async def _monitor_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None

        async def _check_stream(
            stream: asyncio.StreamReader,
            patterns: dict[re.Pattern[str], bool],
        ) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                for pattern in patterns:
                    if pattern.search(decoded):
                        patterns[pattern] = True

        await asyncio.gather(
            _check_stream(self.process.stdout, self._stdout_patterns_found),
            _check_stream(self.process.stderr, self._stderr_patterns_found),
        )

    async def _wait_for_conditions(self) -> None:
        async def check_all() -> bool:
            # Check HTTP endpoints
            http_results = await asyncio.gather(
                *(self._check_http(url) for url in self.wait_http)
            )
            if not all(http_results):
                return False

            # Check TCP ports
            tcp_results = await asyncio.gather(
                *(self._check_tcp(h, p) for h, p in self.wait_tcp)
            )
            if not all(tcp_results):
                return False

            # Check predicates
            pred_results = await asyncio.gather(
                *(self._check_predicate(p) for p in self.wait_predicates)
            )
            if not all(pred_results):
                return False

            # Check output patterns
            if not all(self._stdout_patterns_found.values()):
                return False
            return all(self._stderr_patterns_found.values())

        start_time = asyncio.get_event_loop().time()
        while True:
            if await check_all():
                return

            if (asyncio.get_event_loop().time() - start_time) > self.wait_timeout:
                msg = "Timeout waiting for conditions"
                raise TimeoutError(msg)

            await asyncio.sleep(self.poll_interval)

    async def __aenter__(self) -> Self:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        has_conditions = (
            self.wait_http
            or self.wait_tcp
            or self.wait_predicates
            or self.wait_output
            or self.wait_stderr
        )

        if has_conditions:
            monitor_task = None
            if self.wait_output or self.wait_stderr:
                monitor_task = asyncio.create_task(self._monitor_output())

            try:
                await self._wait_for_conditions()
            except Exception as e:
                self.process.kill()
                if monitor_task:
                    monitor_task.cancel()
                msg = "Failed waiting for process to be ready"
                raise RuntimeError(msg) from e

        return self

    async def __aexit__(self, *_) -> None:
        if self.process:
            self.process.kill()
            await self.process.wait()

    @property
    def pid(self) -> int | None:
        """Return process ID if running."""
        return self.process.pid if self.process else None

    @property
    def returncode(self) -> int | None:
        """Return exit code if process has finished."""
        return self.process.returncode if self.process else None

    def send_signal(self, sig: int) -> None:
        """Send a signal to the process."""
        if self.process:
            self.process.send_signal(sig)


if __name__ == "__main__":
    import asyncio

    from docler.vector_db.dbs.chroma_db import ChromaBackend

    async def main():
        async with ChromaBackend.run_server("./chroma") as runner:
            print(f"Process running with PID {runner.pid}")
        print("finished!")

    asyncio.run(main())
