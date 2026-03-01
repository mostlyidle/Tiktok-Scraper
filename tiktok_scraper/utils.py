"""Helpers: video ID extraction, stealth browser setup, rate limiting."""

import asyncio
import logging
import random
import re
import shutil
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERN = re.compile(r"/video/(\d{15,25})")
_POST_ID_PATTERN = re.compile(r"/(?:video|photo)/(\d{15,25})")


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
        if path.is_file() and path.stem.isdigit():
            ids.add(path.stem)
        elif path.is_dir() and path.name.isdigit():
            ids.add(path.name)
    return ids


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
