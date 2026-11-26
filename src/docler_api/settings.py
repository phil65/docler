"""Server settings of the Docler API, read from the environment."""

from dataclasses import dataclass, field
import ipaddress
import os


TOKEN_ENV = "DOCLER_API_TOKEN"
CORS_ORIGINS_ENV = "DOCLER_API_CORS_ORIGINS"
MAX_UPLOAD_MB_ENV = "DOCLER_API_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 100


@dataclass(frozen=True)
class ApiSettings:
    """Settings of the API server."""

    token: str | None = None
    """Bearer token clients must send (`Authorization: Bearer <token>`)."""

    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    """Origins allowed to call the API from a browser; none (CORS off) by default."""

    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    """Largest request body accepted; bigger ones get 413."""

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Read the settings from `DOCLER_API_*` environment variables."""
        origins = os.environ.get(CORS_ORIGINS_ENV, "")
        max_mb = float(os.environ.get(MAX_UPLOAD_MB_ENV) or DEFAULT_MAX_UPLOAD_MB)
        return cls(
            token=os.environ.get(TOKEN_ENV) or None,
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            max_upload_bytes=int(max_mb * 1024 * 1024),
        )


def is_loopback(host: str | None) -> bool:
    """Whether a host name or address only accepts local connections."""
    if host is None or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind(host: str, settings: ApiSettings) -> None:
    """Refuse to serve on a non-loopback host without a token.

    Raises:
        RuntimeError: If `host` isn't a loopback address and no token is set.
    """
    if not settings.token and not is_loopback(host):
        msg = f"Set {TOKEN_ENV} to serve the Docler API on {host!r} (not a loopback address)"
        raise RuntimeError(msg)
