"""Command line interface for Docler document converter."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import typer


cli = typer.Typer(help="Docler document converter CLI", no_args_is_help=True)

logger = logging.getLogger(__name__)


@cli.command()
def serve() -> None:
    """Start the Streamlit web interface."""
    package_dir = os.path.dirname(__file__)  # noqa: PTH120
    app_path = os.path.join(package_dir, "streamlit_app.py")  # noqa: PTH118
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
