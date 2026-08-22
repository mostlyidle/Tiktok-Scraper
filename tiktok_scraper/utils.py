"""Helpers: video ID extraction, stealth browser setup, rate limiting."""

import asyncio
import datetime as dt
import email.utils
import logging
import random
import re
import shutil
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERN = re.compile(r"/video/(\d{15,25})")
_POST_ID_PATTERN = re.compile(r"/(?:video|photo)/(\d{15,25})")
_TEMPORARY_SUFFIXES = (".part", ".ytdl", ".tmp")


def extract_video_id(url: str) -> str | None:
    """Extract the numeric video ID from a TikTok video URL.

    Args:
        url: A TikTok video URL like
            ``https://www.tiktok.com/@user/video/7123456789012345678``.

    Returns:
        The video ID string, or ``None`` if the URL does not match.
    """
    match = _VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


def extract_post_id(url: str) -> str | None:
    """Extract the numeric post ID from a TikTok video or photo URL.

    Recognises both ``/video/<id>`` and ``/photo/<id>`` URL patterns.

    Args:
        url: A TikTok post URL like
            ``https://www.tiktok.com/@user/video/7123456789012345678``
            or ``https://www.tiktok.com/@user/photo/7123456789012345678``.

    Returns:
        The post ID string, or ``None`` if the URL does not match.
    """
    match = _POST_ID_PATTERN.search(url)
    return match.group(1) if match else None


def get_downloaded_ids(output_dir: Path) -> set[str]:
    """Scan *output_dir* and return the set of already-downloaded video IDs.

    Video files are expected to be named ``<video_id>.<ext>``.
    Photo slideshows are stored as directories named ``<video_id>/``.
    """
    if not output_dir.exists():
        return set()
    ids: set[str] = set()
    for path in output_dir.iterdir():
        if is_completed_artifact(path) and path.stem.isdigit():
            ids.add(path.stem)
        elif is_completed_artifact(path) and path.name.isdigit():
            ids.add(path.name)
    return ids


def is_completed_artifact(path: Path) -> bool:
    """Return whether *path* is a nonempty completed file or directory."""
    if path.is_file():
        return (
            not path.name.lower().endswith(_TEMPORARY_SUFFIXES)
            and path.stat().st_size > 0
        )
    if path.is_dir():
        return any(is_completed_artifact(child) for child in path.iterdir())
    return False


def cleanup_temporary_artifacts(output_dir: Path) -> None:
    """Remove stale yt-dlp/image temporary files without touching downloads.

    Slideshow ``.<id>.partial`` directories are intentionally retained: they
    are the long-standing staging location and are safe because only the
    numeric directory is considered complete.
    """
    if not output_dir.exists():
        return
    for staging_dir in output_dir.iterdir():
        if not staging_dir.is_dir() or not re.fullmatch(
            r"\.\d{15,25}\.partial", staging_dir.name
        ):
            continue
        for path in staging_dir.rglob("*"):
            if path.is_file() and path.name.lower().endswith(_TEMPORARY_SUFFIXES):
                try:
                    path.unlink()
                except OSError:
                    logger.debug("Could not remove stale temporary file %s", path)


def parse_retry_after(
    value: str | None, *, now: dt.datetime | None = None
) -> float | None:
    """Parse a Retry-After seconds value or HTTP date."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        current = now or dt.datetime.now(dt.UTC)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=dt.UTC)
        seconds = (retry_at - current).total_seconds()
    return max(0.0, seconds)


def bounded_backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
    retry_after: float | None = None,
) -> float:
    """Return capped exponential backoff with bounded positive jitter."""
    exponential = min(cap, base * (2 ** max(0, attempt - 1)))
    floor = exponential if retry_after is None else max(exponential, retry_after)
    jitter = float(random.uniform(0.0, min(1.0, max(0.0, floor * 0.25))))
    return float(min(cap, floor + jitter))


async def random_delay(min_seconds: float, max_seconds: float) -> None:
    """Sleep for a random duration between *min_seconds* and *max_seconds*."""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)


async def create_stealth_browser_context(
    headless: bool = True,
    cookies_file: Path | None = None,
) -> tuple[Playwright, Browser, BrowserContext]:
    """Create a Playwright browser context with stealth-like settings.

    Returns a ``(playwright_instance, browser, browser_context)`` tuple.
    The caller is responsible for closing all three when done::

        pw, browser, context = await create_stealth_browser_context()
        try:
            ...
        finally:
            await context.close()
            await browser.close()
            await pw.stop()

    Args:
        headless: Whether to run the browser in headless mode.
        cookies_file: Optional path to a Netscape-format cookies file.

    Returns:
        Tuple of (playwright instance, browser, browser context).
    """
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )

    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
    )

    # Remove the webdriver navigator property that flags automation.
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
    """)

    if cookies_file is not None:
        await _load_cookies(context, cookies_file)

    return pw, browser, context


async def _load_cookies(context: BrowserContext, cookies_file: Path) -> None:
    """Load Netscape-format cookies into the browser context."""
    if not cookies_file.exists():
        logger.warning("Cookies file %s does not exist; skipping.", cookies_file)
        return

    cookies: list[dict[str, object]] = []
    with open(cookies_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            domain, _, path, secure, expires, name, value = parts[:7]
            cookies.append(
                {
                    "name": name,
                    "value": value,
                    "domain": domain,
                    "path": path,
                    "secure": secure.upper() == "TRUE",
                    "expires": int(expires) if expires.isdigit() else -1,
                }
            )

    if cookies:
        await context.add_cookies(cookies)  # type: ignore[arg-type]
        logger.info("Loaded %d cookies from %s.", len(cookies), cookies_file)


def check_ffmpeg() -> bool:
    """Check whether ffmpeg is available on the system PATH.

    Logs a warning if ffmpeg is not found, since it is required for
    merging separate video and audio streams downloaded by yt-dlp.

    Returns:
        ``True`` if ffmpeg is found, ``False`` otherwise.
    """
    if shutil.which("ffmpeg") is not None:
        return True

    logger.warning(
        "ffmpeg not found on PATH. ffmpeg is required to merge video and "
        "audio streams. Downloads may lack audio. "
        "Install ffmpeg: https://ffmpeg.org/download.html",
    )
    return False
