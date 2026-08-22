"""Unit tests for the CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from tiktok_scraper.cli import main
from tiktok_scraper.models import ScrapeResult


@pytest.fixture()
def cli_runner() -> CliRunner:
    """A Click test runner."""
    return CliRunner()


class TestCli:
    """Tests for the CLI entry point."""

    def test_help_flag(self, cli_runner: CliRunner) -> None:
        """--help should print usage and exit 0."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "USERNAME" in result.output

    def test_missing_username(self, cli_runner: CliRunner) -> None:
        """Running without a username should fail."""
        result = cli_runner.invoke(main, [])
        assert result.exit_code != 0

    @pytest.mark.parametrize("username", [".", "..", "bad user", "a/b"])
    def test_rejects_unsafe_username(
        self,
        cli_runner: CliRunner,
        username: str,
    ) -> None:
        result = cli_runner.invoke(main, [username])
        assert result.exit_code != 0

    def test_rejects_negative_limit(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["testuser", "--limit", "-1"])
        assert result.exit_code != 0

    def test_strips_at_sign(self, cli_runner: CliRunner) -> None:
        """Leading @ should be stripped from the username."""
        mock_result = ScrapeResult(
            username="testuser",
            total_found=0,
            downloaded=0,
        )

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            cli_runner.invoke(main, ["@testuser"])

        # asyncio.run should have been called with a coroutine from _run.
        assert mock_asyncio.run.called
        # Verify the config passed to _run has the @ stripped.
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config.username == "testuser"

    def test_custom_output_dir(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """--output-dir should be passed through to the config."""
        mock_result = ScrapeResult(username="testuser", total_found=0)
        out = tmp_path / "custom_output"

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser", "--output-dir", str(out)])

        assert result.exit_code == 0

    def test_limit_option(self, cli_runner: CliRunner) -> None:
        """--limit should be accepted."""
        mock_result = ScrapeResult(username="testuser", total_found=0)

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser", "--limit", "5"])

        assert result.exit_code == 0

    def test_exit_code_on_failures(self, cli_runner: CliRunner) -> None:
        """Non-zero exit code when downloads fail."""
        mock_result = ScrapeResult(
            username="testuser",
            total_found=5,
            downloaded=3,
            failed=2,
            failed_videos=["123", "456"],
        )

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert result.exit_code == 1

    def test_summary_output(self, cli_runner: CliRunner) -> None:
        """Summary should be printed after a successful run."""
        mock_result = ScrapeResult(
            username="testuser",
            total_found=10,
            downloaded=8,
            skipped=2,
            failed=0,
        )

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Scrape complete" in result.output
        assert "Videos downloaded: 8" in result.output
        assert "Skipped (exists):  2" in result.output
        assert result.exit_code == 0

    def test_summary_shows_audio_only_skipped(self, cli_runner: CliRunner) -> None:
        """Summary should show audio-only skipped count when non-zero."""
        mock_result = ScrapeResult(
            username="testuser",
            total_found=10,
            downloaded=6,
            photos_downloaded=1,
            skipped=1,
            skipped_audio_only=2,
            failed=0,
        )

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Skipped (audio):   2" in result.output
        assert "Skipped (exists):  1" in result.output
        assert result.exit_code == 0

    def test_summary_hides_audio_only_when_zero(self, cli_runner: CliRunner) -> None:
        """Summary should not show audio-only line when count is zero."""
        mock_result = ScrapeResult(
            username="testuser",
            total_found=5,
            downloaded=4,
            skipped=1,
            failed=0,
        )

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock),
            patch("tiktok_scraper.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Skipped (audio)" not in result.output
        assert result.exit_code == 0
