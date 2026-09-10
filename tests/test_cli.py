"""Unit tests for the CLI module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from tiktok_scraper.cli import _setup_logging, main
from tiktok_scraper.exceptions import CreatorNotFoundError
from tiktok_scraper.models import ScrapeResult


@pytest.fixture()
def cli_runner() -> CliRunner:
    """A Click test runner."""
    return CliRunner()


class TestCli:
    """Tests for the CLI entry point."""

    def test_logging_setup_is_reentrant(self) -> None:
        """Repeated setup should update verbosity without adding handlers."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        root_logger.handlers.clear()

        try:
            _setup_logging(False)
            _setup_logging(True)

            cli_handlers = [
                handler
                for handler in root_logger.handlers
                if handler.get_name() == "tiktok_scraper.cli"
            ]
            assert len(cli_handlers) == 1
            assert root_logger.level == logging.DEBUG
        finally:
            for handler in root_logger.handlers:
                handler.close()
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)

    def test_help_flag(self, cli_runner: CliRunner) -> None:
        """--help should print usage and exit 0."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "USERNAME" in result.output

    def test_version_flag(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

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

    def test_rejects_cookie_directory(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        cookies = tmp_path / "cookies"
        cookies.mkdir()

        result = cli_runner.invoke(main, ["testuser", "--cookies", str(cookies)])

        assert result.exit_code != 0

    def test_rejects_output_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        output_file = tmp_path / "output"
        output_file.write_text("not a directory", encoding="utf-8")

        result = cli_runner.invoke(
            main,
            ["testuser", "--output-dir", str(output_file)],
        )

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
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["@testuser"])

        assert result.exit_code == 0
        # Verify the config passed to _run has the @ stripped.
        mock_run.assert_awaited_once()
        config = mock_run.call_args[0][0]
        assert config.username == "testuser"

    def test_custom_output_dir(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """--output-dir should be passed through to the config."""
        mock_result = ScrapeResult(username="testuser", total_found=0)
        out = tmp_path / "custom_output"

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser", "--output-dir", str(out)])

        assert result.exit_code == 0
        mock_run.assert_awaited_once()
        config = mock_run.call_args[0][0]
        assert config.output_dir == out

    def test_options_are_propagated(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """CLI options should be copied into the scraper configuration."""
        mock_result = ScrapeResult(username="testuser", total_found=0)
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        output_dir = tmp_path / "output"

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(
                main,
                [
                    "testuser",
                    "--output-dir",
                    str(output_dir),
                    "--cookies",
                    str(cookies),
                    "--limit",
                    "5",
                    "--no-headless",
                    "--concurrency",
                    "7",
                    "--video-only",
                ],
            )

        assert result.exit_code == 0
        mock_run.assert_awaited_once()
        config = mock_run.call_args[0][0]
        assert config.output_dir == output_dir
        assert config.cookies_file == cookies
        assert config.limit == 5
        assert config.headless is False
        assert config.max_concurrent_downloads == 7
        assert config.video_only is True

    def test_limit_zero_skips_discovery_and_download(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """A zero limit should finish without touching TikTok or the output."""
        output_dir = tmp_path / "output"

        with (
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
            patch(
                "tiktok_scraper.cli.fetch_video_urls",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch(
                "tiktok_scraper.cli.download_videos",
                new_callable=AsyncMock,
            ) as mock_download,
        ):
            result = cli_runner.invoke(
                main,
                ["testuser", "--limit", "0", "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0
        mock_run.assert_not_awaited()
        mock_fetch.assert_not_awaited()
        mock_download.assert_not_awaited()
        assert not output_dir.exists()
        assert "Posts found:       0" in result.output

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
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
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
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Scrape complete" in result.output
        assert "Videos downloaded:      8" in result.output
        assert "Skipped (already present): 2" in result.output
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
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Skipped (audio):   2" in result.output
        assert "Skipped (already present): 1" in result.output
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
            patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Skipped (audio)" not in result.output
        assert result.exit_code == 0

    def test_summary_reports_audio_posts_separately(self, cli_runner: CliRunner) -> None:
        mock_result = ScrapeResult(
            username="testuser",
            total_found=2,
            audio_downloaded=1,
        )

        with patch("tiktok_scraper.cli._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = cli_runner.invoke(main, ["testuser"])

        assert "Audio downloaded:       1" in result.output
        assert "Videos downloaded:      0" in result.output

    def test_scraper_error_is_reported(self, cli_runner: CliRunner) -> None:
        """Scraper failures should produce a normal Click error message."""
        with patch(
            "tiktok_scraper.cli._run",
            new_callable=AsyncMock,
            side_effect=CreatorNotFoundError("missing"),
        ) as mock_run:
            result = cli_runner.invoke(main, ["missing"])

        assert result.exit_code == 1
        assert "Error: TikTok creator @missing not found." in result.output
        mock_run.assert_awaited_once()
