"""Main entry point for Docler API server."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

from docler.log import get_logger


cli = typer.Typer(help="Docler API server", no_args_is_help=True)
logger = get_logger(__name__)


@cli.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    log_level: str = typer.Option("info", help="Log level"),
    reload: bool = typer.Option(False, help="Enable auto-reload on file changes"),
):
    """Start the Docler API server."""
    # Add the parent directory to sys.path to ensure imports work correctly
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import uvicorn

    try:
        uvicorn.run(
            "docler_api.main:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
