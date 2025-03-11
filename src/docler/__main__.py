"""Command line interface for Docler document converter."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys

import typer


cli = typer.Typer(help="Docler document converter CLI", no_args_is_help=True)

logger = logging.getLogger(__name__)

PACKAGE_DIR = pathlib.Path(__file__).parent


@cli.command()
def serve() -> None:
    """Start the Streamlit web interface."""
    app_path = str(PACKAGE_DIR / "streamlit_app.py")
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except subprocess.CalledProcessError as e:
        # msg = f"Failed to start Streamlit: {e}"
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@cli.command("chunk_ui")
def chunk_ui() -> None:
    """Start the Streamlit web interface."""
    app_path = str(PACKAGE_DIR / "streamlit_chunk_app.py")
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except subprocess.CalledProcessError as e:
        # msg = f"Failed to start Streamlit: {e}"
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        logger.info("Shutting down...")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
