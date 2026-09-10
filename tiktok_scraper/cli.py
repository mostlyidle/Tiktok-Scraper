"""CLI entry point for the TikTok scraper."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from tiktok_scraper.config import ScraperConfig, normalize_username
from tiktok_scraper.downloader import download_videos
from tiktok_scraper.exceptions import CreatorNotFoundError, ScraperError
from tiktok_scraper.models import ScrapeResult
from tiktok_scraper.scraper import fetch_video_urls

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
_CLI_HANDLER_NAME = "tiktok_scraper.cli"


def _setup_logging(verbose: bool) -> None:
    """Configure console logging without duplicating handlers on re-entry."""
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if handler.get_name() == _CLI_HANDLER_NAME
        ),
        None,
    )

    # Respect an application's existing logging configuration.  When the CLI
    # owns the console handler, update it so repeated invocations follow the
    # current verbosity and Click's current stderr stream.
    if console_handler is None and not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.set_name(_CLI_HANDLER_NAME)
        root_logger.addHandler(console_handler)

    if isinstance(console_handler, logging.StreamHandler):
        console_handler.setLevel(logging.NOTSET)
        console_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        )
        if console_handler.stream is not sys.stderr:
            console_handler.setStream(sys.stderr)

    # Silence noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def _username(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    """Click callback that validates USERNAME before starting any work."""
    try:
        return normalize_username(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="USERNAME") from exc


def _regular_file(
    _ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Ensure an existing path is a regular file."""
    if value is not None and not value.is_file():
        raise click.BadParameter("must be a regular file", param=param)
    return value


def _directory_compatible(
    _ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Allow an existing directory or a path whose parents can be created."""
    if value is None:
        return None

    if value.is_symlink() and not value.exists():
        raise click.BadParameter(
            "must be a directory or a new directory path", param=param
        )
    if value.exists() and not value.is_dir():
        raise click.BadParameter(
            "must be a directory or a new directory path", param=param
        )

    parent = value.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise click.BadParameter("must have a directory-compatible parent", param=param)
    return value


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="tiktok-scraper")
@click.argument("username", callback=_username)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    callback=_directory_compatible,
    help="Directory to save videos (default: ./downloads/<username>/).",
)
@click.option(
    "--cookies",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    callback=_regular_file,
    help="Path to a Netscape-format cookies file for authentication.",
)
@click.option(
    "-n",
    "--limit",
    type=click.IntRange(min=0),
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

    try:
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
    except ValueError as exc:
        raise click.ClickException(f"Invalid configuration: {exc}") from exc

    logger.info("Starting scrape for @%s", username)
    logger.info("Output directory: %s", config.output_dir)

    if config.limit == 0:
        logger.info("Limit is 0; skipping discovery and downloads.")
        _print_summary(ScrapeResult(username=config.username))
        return

    try:
        result = asyncio.run(_run(config))
    except CreatorNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except ScraperError as exc:
        raise click.ClickException(f"Scraper error: {exc}") from exc
    except OSError as exc:
        raise click.ClickException(
            f"Could not access output directory {config.output_dir}: {exc}"
        ) from exc
    except ValueError as exc:
        raise click.ClickException(f"Invalid scraper data: {exc}") from exc
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        raise SystemExit(130) from None

    _print_summary(result)

    if result.failed > 0:
        raise SystemExit(1)


def _print_summary(result: ScrapeResult) -> None:
    """Print the result counts using labels that match their semantics."""
    click.echo()
    click.echo(f"Scrape complete for @{result.username}")
    click.echo(f"  Posts found:       {result.total_found}")
    click.echo(f"  Videos downloaded:      {result.downloaded}")
    if result.audio_downloaded:
        click.echo(f"  Audio downloaded:       {result.audio_downloaded}")
    click.echo(f"  Photo posts downloaded: {result.photos_downloaded}")
    click.echo(f"  Skipped (already present): {result.skipped}")
    if result.skipped_audio_only:
        click.echo(f"  Skipped (audio):   {result.skipped_audio_only}")
    click.echo(f"  Failed:            {result.failed}")
    if result.failed_videos:
        click.echo(f"  Failed IDs:        {', '.join(result.failed_videos)}")
    for video_id, reason in result.failed_reasons.items():
        click.echo(f"  Failure reason:    {video_id}: {reason}")


async def _run(config: ScraperConfig) -> ScrapeResult:
    """Async entry point: discover videos then download them.

    Discovery uses yt-dlp first and starts a Playwright browser only when
    yt-dlp cannot discover posts.
    """
    if config.limit == 0:
        return ScrapeResult(username=config.username)

    videos = await fetch_video_urls(config)

    if not videos:
        logging.getLogger(__name__).warning("No videos found for @%s.", config.username)
        return ScrapeResult(username=config.username)

    result = await download_videos(videos, config)
    return result
