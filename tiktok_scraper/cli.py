"""CLI entry point for the TikTok scraper."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.downloader import download_videos
from tiktok_scraper.exceptions import CreatorNotFoundError, ScraperError
from tiktok_scraper.models import ScrapeResult
from tiktok_scraper.scraper import fetch_video_urls
from tiktok_scraper.utils import check_ffmpeg


def _setup_logging(verbose: bool) -> None:
    """Configure the root logger for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Silence noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("username")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save videos (default: ./downloads/<username>/).",
)
@click.option(
    "--cookies",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a Netscape-format cookies file for authentication.",
)
@click.option(
    "-n",
    "--limit",
    type=int,
    default=None,
    help="Maximum number of videos to download.",
)
@click.option(
    "--headless/--no-headless",
    default=True,
    help="Run browser in headless mode (default: headless).",
)
@click.option(
    "-c",
    "--concurrency",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Number of concurrent downloads.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
@click.option(
    "--video-only/--no-video-only",
    default=False,
    help="Skip audio-only posts that have no video or photo content.",
)
def main(
    username: str,
    output_dir: Path | None,
    cookies: Path | None,
    limit: int | None,
    headless: bool,
    concurrency: int,
    verbose: bool,
    video_only: bool,
) -> None:
    """Scrape all videos from a TikTok creator's profile.

    USERNAME is the TikTok handle (without the leading @).
    """
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    check_ffmpeg()

    # Strip leading @ if the user included it.
    username = username.removeprefix("@")

    if not username:
        raise click.BadParameter("Username must not be empty.", param_hint="USERNAME")

    if output_dir is None:
        output_dir = ScraperConfig.default_output_dir(username)

    config = ScraperConfig(
        username=username,
        output_dir=output_dir,
        cookies_file=cookies,
        limit=limit,
        headless=headless,
        max_concurrent_downloads=concurrency,
        video_only=video_only,
    )

    logger.info("Starting scrape for @%s", username)
    logger.info("Output directory: %s", config.output_dir)

    try:
        result = asyncio.run(_run(config))
    except CreatorNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except ScraperError as exc:
        logger.error("Scraper error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        raise SystemExit(130) from None

    # Print summary.
    click.echo()
    click.echo(f"Scrape complete for @{result.username}")
    click.echo(f"  Posts found:       {result.total_found}")
    click.echo(f"  Videos downloaded: {result.downloaded}")
    click.echo(f"  Photos downloaded: {result.photos_downloaded}")
    click.echo(f"  Skipped (exists):  {result.skipped}")
    if result.skipped_audio_only:
        click.echo(f"  Skipped (audio):   {result.skipped_audio_only}")
    click.echo(f"  Failed:            {result.failed}")
    if result.failed_videos:
        click.echo(f"  Failed IDs:        {', '.join(result.failed_videos)}")

    if result.failed > 0:
        raise SystemExit(1)


async def _run(config: ScraperConfig) -> ScrapeResult:
    """Async entry point: discover videos then download them.

    Discovery uses yt-dlp first and starts a Playwright browser only when
    yt-dlp cannot discover posts.
    """
    videos = await fetch_video_urls(config)

    if not videos:
        logging.getLogger(__name__).warning("No videos found for @%s.", config.username)
        return ScrapeResult(username=config.username)

    result = await download_videos(videos, config)
    return result
