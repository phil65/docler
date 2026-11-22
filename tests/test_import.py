"""Smoke test: the package imports and reports its version."""

from __future__ import annotations

import docler


def test_package_imports() -> None:
    """The package can be imported and exposes a version string."""
    assert isinstance(docler.__version__, str)
