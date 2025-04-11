"""Process runner with configurable wait conditions."""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import os
import re
import socket
import subprocess
import threading
from typing import IO, TYPE_CHECKING, Literal, Self
import urllib.parse
import weakref


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

RegexPattern = str

_process_registry: dict[tuple[str, ...], weakref.ref[subprocess.Popen]] = {}
_registry_lock = asyncio.Lock()


def _clean_registry():
    """Remove dead processes from registry."""
    dead_keys = []
    for key, ref in _process_registry.items():
        process = ref()
        if process is None or process.poll() is not None:
            dead_keys.append(key)

    for key in dead_keys:
        del _process_registry[key]


async def _check_http(url: str) -> bool:
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


async def _check_tcp(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()
    except (ConnectionRefusedError, socket.gaierror):
        return False
    else:
        return True


async def _check_predicate(
    pred: Callable[[], bool] | Callable[[], Awaitable[bool]],
) -> bool:
    if asyncio.iscoroutinefunction(pred):
        return await pred()
    return await asyncio.to_thread(pred)  # type: ignore


def _reader_thread(
    stream: IO[bytes],
    buffer: list[str],
    patterns: dict[re.Pattern[str], bool] | None = None,
):
    """Thread to read from process streams."""
    try:
        for line in iter(stream.readline, b""):
            try:
                decoded = line.decode(errors="ignore").rstrip()
                buffer.append(decoded)  # Store the line in the buffer
                if patterns:
                    for pattern in patterns:
                        if pattern.search(decoded):
                            patterns[pattern] = True
            except Exception:  # noqa: BLE001
                continue  # Skip any decoding errors
    except ValueError:
        # Stream likely closed
        pass


class ProcessRunner:
    def __init__(
        self,
        command: list[str] | str,
        *,
        reuse: bool = False,
        wait_http: list[str] | None = None,  # ["http://localhost:8000/health"]
        wait_tcp: list[tuple[str, int]] | None = None,  # [("localhost", 6379)]
        wait_predicates: list[Callable[[], bool] | Callable[[], Awaitable[bool]]]
        | None = None,
        wait_output: list[RegexPattern] | None = None,
        wait_stderr: list[RegexPattern] | None = None,
        wait_timeout: float = 30.0,
        poll_interval: float = 0.1,
        cleanup_timeout: float = 5.0,
    ):
        self.cleanup_timeout = cleanup_timeout
        self.command = command if isinstance(command, list) else command.split()
        self.command_key = tuple(self.command)
        self.reuse = reuse
        self.wait_http = wait_http or []
        self.wait_tcp = wait_tcp or []
        self.wait_predicates = wait_predicates or []
        self.wait_output = [re.compile(p) for p in (wait_output or [])]
        self.wait_stderr = [re.compile(p) for p in (wait_stderr or [])]
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.process: subprocess.Popen | None = None
        self._stdout_patterns_found = dict.fromkeys(self.wait_output, False)
        self._stderr_patterns_found = dict.fromkeys(self.wait_stderr, False)
        self._stdout_buffer: list[str] = []
        self._stderr_buffer: list[str] = []
        # Threads for stream reading
        self._stdout_reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None

    async def _wait_for_conditions(self):
        async def check_all() -> bool:
            # Check HTTP endpoints
            http_results = await asyncio.gather(
                *(_check_http(url) for url in self.wait_http)
            )
            if not all(http_results):
                return False

            # Check TCP ports
            tcp_results = await asyncio.gather(
                *(_check_tcp(h, p) for h, p in self.wait_tcp)
            )
            if not all(tcp_results):
                return False

            # Check predicates
            pred_results = await asyncio.gather(
                *(_check_predicate(p) for p in self.wait_predicates)
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
        if self.reuse:
            async with _registry_lock:
                _clean_registry()
                if self.command_key in _process_registry:
                    existing_process = _process_registry[self.command_key]()
                    if existing_process is not None and existing_process.poll() is None:
                        self.process = existing_process
                        return self

        # Use shell=True for Windows and non-standard command execution
        is_windows = os.name == "nt"
        # For Windows, use the raw command string with shell=True to avoid issues
        if is_windows and isinstance(self.command, list):
            cmd_str = subprocess.list2cmdline(self.command)
        else:
            cmd_str = (
                self.command
                if isinstance(self.command, str)
                else subprocess.list2cmdline(self.command)
            )

        self.process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=is_windows,  # Use shell on Windows
            text=False,  # Keep as bytes for proper line handling
            bufsize=0,  # Unbuffered
        )

        _process_registry[self.command_key] = weakref.ref(self.process)

        # Start reader threads
        self._stdout_reader = threading.Thread(
            target=_reader_thread,
            args=(self.process.stdout, self._stdout_buffer, self._stdout_patterns_found),
            daemon=True,
        )
        self._stdout_reader.start()

        self._stderr_reader = threading.Thread(
            target=_reader_thread,
            args=(self.process.stderr, self._stderr_buffer, self._stderr_patterns_found),
            daemon=True,
        )
        self._stderr_reader.start()

        has_conditions = (
            self.wait_http
            or self.wait_tcp
            or self.wait_predicates
            or self.wait_output
            or self.wait_stderr
        )

        if has_conditions:
            try:
                await self._wait_for_conditions()
            except Exception as e:
                self.process.kill()
                msg = "Failed waiting for process to be ready"
                raise RuntimeError(msg) from e

        return self

    async def __aexit__(self, *_):
        """Clean up process with timeout."""
        if self.process is None:
            return
        try:
            self.process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=self.cleanup_timeout)
        except ProcessLookupError:
            pass

    @property
    def pid(self) -> int | None:
        """Return process ID if running."""
        return self.process.pid if self.process else None

    @property
    def returncode(self) -> int | None:
        """Return exit code if process has finished."""
        return self.process.poll() if self.process else None

    def send_signal(self, sig: int):
        """Send a signal to the process."""
        if self.process:
            self.process.send_signal(sig)

    async def get_output(self, stream: Literal["stdout", "stderr"] = "stdout") -> str:
        """Get current output content from stdout or stderr."""
        buffer = self._stderr_buffer if stream == "stderr" else self._stdout_buffer
        return "\n".join(buffer)


if __name__ == "__main__":

    async def main():
        docker_cmd = "docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
        print(f"Running command: {docker_cmd}")
        async with ProcessRunner(docker_cmd, wait_tcp=[("localhost", 6333)]) as runner:
            print(f"Process running with PID {runner.pid}")
            print("\nSTDOUT:")
            print(await runner.get_output())
            print("\nSTDERR:")
            print(await runner.get_output("stderr"))

        print("Process completed")

    asyncio.run(main())
