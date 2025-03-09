"""Command line interface for Docler document converter."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Annotated

import typer

from docler.lang_code import SupportedLanguage  # noqa: TC001


cli = typer.Typer(
    help="Docler document converter CLI",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


@cli.command()
def serve(
    language: Annotated[
        SupportedLanguage,
        typer.Option(help="Primary language for OCR/processing"),
    ] = "en",
    port: Annotated[
        int,
        typer.Option(help="Port to run Streamlit on"),
    ] = 8501,
) -> None:
    """Start the Streamlit web interface."""
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        # msg = (
        #     "Streamlit is required for the web interface. "
        #     "Install it with: pip install docler[streamlit]"
        # )
        raise typer.Exit(1) from e

    # Get the path to streamlit_app.py
    package_dir = os.path.dirname(__file__)  # noqa: PTH120
    app_path = os.path.join(package_dir, "streamlit_app.py")  # noqa: PTH118

    # Set environment variables for the app
    env = os.environ.copy()
    env["DOCLER_LANGUAGE"] = language

    try:
        # Launch streamlit
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                "--server.port",
                str(port),
            ],
            env=env,
            check=True,
        )
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
