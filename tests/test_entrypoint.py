"""Tests for the package module entry point."""

from __future__ import annotations

import runpy
from unittest.mock import patch


def test_module_entrypoint_invokes_cli_once() -> None:
    """``python -m tiktok_scraper`` should dispatch to the CLI once."""
    with patch("tiktok_scraper.cli.main") as mock_main:
        runpy.run_module("tiktok_scraper.__main__", run_name="__main__")

    mock_main.assert_called_once_with()
