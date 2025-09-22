"""Command line interface for Docler document converter."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import typer

from docler.log import get_logger


cli = typer.Typer(help="Docler document converter CLI", no_args_is_help=True)

logger = get_logger(__name__)

PACKAGE_DIR = pathlib.Path(__file__).parent.parent / "docler_streamlit"


@cli.command()
def flow():
    """Start the Streamlit web interface."""
    app_path = str(PACKAGE_DIR / "app.py")
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except subprocess.CalledProcessError as e:
        # msg = f"Failed to start Streamlit: {e}"
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@cli.command("chunk_ui")
def chunk_ui():
    """Start the Streamlit web interface."""
    app_path = str(PACKAGE_DIR / "chunk_app.py")
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except subprocess.CalledProcessError as e:
        # msg = f"Failed to start Streamlit: {e}"
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@cli.command("ocr_ui")
def ocr_ui():
    """Start the Streamlit web interface."""
    app_path = str(PACKAGE_DIR / "ocr_app.py")
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except subprocess.CalledProcessError as e:
        # msg = f"Failed to start Streamlit: {e}"
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@cli.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    log_level: str = typer.Option("info", help="Log level"),
    reload: bool = typer.Option(False, help="Enable auto-reload on file changes"),
):
    """Start the Docler API server."""
    try:
        # Lazy import to avoid unnecessary dependencies if not using the API
        import uvicorn

        from docler_api.main import app

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
        )
    except ImportError as e:
        msg = f"Failed to import API components: {e}. Is the 'server' extra installed?"
        logger.exception(msg)
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@cli.command()
def mcp(
    transport: str = typer.Option("stdio", help="Transport protocol to use"),
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
):
    """Start the MCP server."""
    # Validate transport choice
    valid_transports = ["stdio", "sse", "streamable-http"]
    if transport not in valid_transports:
        typer.echo(
            f"Error: Invalid transport '{transport}'. Choose from: {', '.join(valid_transports)}"
        )
        raise typer.Exit(1)

    try:
        # Lazy import to avoid unnecessary dependencies if not using MCP
        from docler_mcp.server import mcp as mcp_server

        if transport == "stdio":
            mcp_server.run(transport=transport)
        else:
            mcp_server.run(transport=transport, host=host, port=port)
    except ImportError as e:
        msg = f"Failed to import MCP components: {e}. Is the 'mcp' extra installed?"
        logger.exception(msg)
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
