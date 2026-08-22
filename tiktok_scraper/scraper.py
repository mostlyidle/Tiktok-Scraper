"""Core scraping logic — discover video URLs for a TikTok creator.

Primary strategy uses yt-dlp's ``TikTokUser`` extractor which handles
TikTok's anti-bot protections natively.  An optional Playwright-based
fallback (browser scrolling + DOM parsing + API interception) is
retained for situations where yt-dlp extraction fails.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

import yt_dlp
from playwright.async_api import BrowserContext, Page, Response, TimeoutError
from yt_dlp.utils import DownloadError as YtdlpDownloadError

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.exceptions import BrowserError, CreatorNotFoundError
from tiktok_scraper.models import VideoInfo
from tiktok_scraper.utils import (
    bounded_backoff,
    create_stealth_browser_context,
    extract_post_id,
    parse_retry_after,
    random_delay,
)

logger = logging.getLogger(__name__)

# Maximum consecutive scrolls with no new videos before we stop (browser fallback).
_MAX_STALE_SCROLLS = 5

# Selectors / URL fragments used to identify video links (browser fallback).
_VIDEO_LINK_SELECTOR = 'a[href*="/video/"]'
_PHOTO_LINK_SELECTOR = 'a[href*="/photo/"]'
_API_ITEM_LIST_FRAGMENT = "/api/post/item_list/"
_PERMANENT_YTDLP_ERRORS = (
    "404",
    "unsupported url",
    "private video",
    "login required",
    "copyright",
    "forbidden",
    "not available",
)
_TRANSIENT_YTDLP_ERRORS = (
    "429",
    "rate limit",
    "rate-limit",
    "too many requests",
    "temporarily",
    "timed out",
    "timeout",
    "connection",
    "network",
    "502",
    "503",
    "504",
    "server error",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_video_urls(
    config: ScraperConfig,
    browser_context: BrowserContext | None = None,
) -> list[VideoInfo]:
    """Discover video URLs for a TikTok creator.

    Tries yt-dlp first (fast, reliable, handles anti-bot natively).
    Falls back to Playwright browser scraping when yt-dlp fails.  A supplied
    *browser_context* is reused; otherwise one is created only for the
    fallback attempt.

    Args:
        config: The scraper configuration (username, limit, etc.).
        browser_context: Optional Playwright browser context to use for the
            browser-based fallback strategy.

    Returns:
        A deduplicated, sorted list of :class:`VideoInfo` objects.

    Raises:
        CreatorNotFoundError: If the profile does not exist.
        BrowserError: If both strategies fail.
    """
    # ---- Strategy 1: yt-dlp (primary) ----
    try:
        videos = await _fetch_via_ytdlp(config)
        if videos:
            return videos
        logger.warning(
            "yt-dlp returned no videos for @%s. Trying browser fallback.",
            config.username,
        )
    except CreatorNotFoundError:
        raise
    except Exception as exc:
        logger.warning(
            "yt-dlp discovery failed for @%s: %s. Trying browser fallback.",
            config.username,
            exc,
        )

    # ---- Strategy 2: Playwright browser (fallback) ----
    if browser_context is not None:
        return await _fetch_via_browser(config, browser_context)

    logger.info("Attempting Playwright browser fallback...")
    try:
        pw, browser, context = await create_stealth_browser_context(
            headless=config.headless,
            cookies_file=config.cookies_file,
        )
    except Exception as exc:
        raise BrowserError(f"Could not start browser fallback: {exc}") from exc

    try:
        return await _fetch_via_browser(config, context)
    finally:
        await context.close()
        await browser.close()
        await pw.stop()


# ---------------------------------------------------------------------------
# Strategy 1 — yt-dlp discovery
# ---------------------------------------------------------------------------


async def _fetch_via_ytdlp(config: ScraperConfig) -> list[VideoInfo]:
    """Use yt-dlp's TikTokUser extractor to list video URLs.

    Args:
        config: Scraper configuration.

    Returns:
        List of :class:`VideoInfo` objects discovered.

    Raises:
        CreatorNotFoundError: If yt-dlp reports the user does not exist.
    """
    profile_url = f"https://www.tiktok.com/@{config.username}"
    logger.info("Discovering videos via yt-dlp for %s", profile_url)

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "sleep_interval_requests": config.request_delay,
        "sleep_interval": config.request_delay,
        "max_sleep_interval": 2 * config.request_delay,
    }

    if config.limit is not None:
        ydl_opts["playlistend"] = config.limit

    if config.cookies_file is not None:
        ydl_opts["cookiefile"] = str(config.cookies_file)

    loop = asyncio.get_running_loop()

    async def _extract() -> dict[str, Any] | None:
        return await loop.run_in_executor(
            None,
            functools.partial(_sync_extract, profile_url, ydl_opts),
        )

    try:
        info = await _retry_ytdlp_operation(_extract, config, "discovery")
    except YtdlpDownloadError as exc:
        if _is_creator_not_found(str(exc)):
            raise CreatorNotFoundError(config.username) from exc
        raise

    if info is None:
        return []

    videos_by_id: dict[str, VideoInfo] = {}
    entries = info.get("entries")
    if entries is None:
        entries = []
    elif isinstance(entries, Iterable):
        entries = list(entries)
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        vid = str(entry.get("id", ""))
        url = entry.get("url", "")
        if not vid or not url:
            continue
        if vid not in videos_by_id:
            videos_by_id[vid] = VideoInfo(video_id=vid, url=url)

    all_videos = sorted(videos_by_id.values(), key=lambda v: v.video_id)

    if config.limit is not None:
        all_videos = all_videos[: config.limit]

    logger.info(
        "yt-dlp discovered %d videos for @%s.",
        len(all_videos),
        config.username,
    )
    return all_videos


def _sync_extract(url: str, opts: dict[str, Any]) -> dict[str, Any] | None:
    """Synchronous yt-dlp extraction (runs inside an executor)."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.extract_info(url, download=False)
        if result is None:
            return None
        info = dict(result)
        # Materialize entries eagerly — yt-dlp may return a lazy generator
        # that becomes invalid after the YoutubeDL context manager exits.
        entries = info.get("entries")
        if entries is None:
            info["entries"] = []
        elif isinstance(entries, Iterable):
            info["entries"] = list(entries)
        else:
            info["entries"] = []
        return info


async def _retry_ytdlp_operation(
    operation: Any,
    config: ScraperConfig,
    label: str,
) -> Any:
    """Run a yt-dlp operation with bounded adaptive retry spacing."""
    attempts = max(1, config.download_retries)
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            message = str(exc).lower()
            if _is_creator_not_found(message) or not _is_retryable_ytdlp(exc):
                raise
            if attempt >= attempts:
                raise
            retry_after = _retry_after_from_message(message)
            delay = bounded_backoff(
                attempt,
                base=config.retry_backoff_base,
                cap=config.retry_backoff_max,
                retry_after=retry_after,
            )
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                label.capitalize(),
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("retry loop must return or raise")


def _is_creator_not_found(message: str) -> bool:
    """Recognise creator-not-found errors without broad transient matches."""
    return any(
        phrase in message
        for phrase in (
            "user does not exist",
            "this account does not exist",
            "creator not found",
            "couldn't find this account",
        )
    )


def _is_retryable_ytdlp(exc: Exception) -> bool:
    """Return whether an yt-dlp failure is likely transient."""
    message = str(exc).lower()
    if _is_creator_not_found(message) or any(
        phrase in message for phrase in _PERMANENT_YTDLP_ERRORS
    ):
        return False
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    return any(phrase in message for phrase in _TRANSIENT_YTDLP_ERRORS)


def _retry_after_from_message(message: str) -> float | None:
    """Extract a seconds hint from yt-dlp's human-readable error text."""
    match = re.search(r"retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)", message)
    return parse_retry_after(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Strategy 2 — Playwright browser (fallback)
# ---------------------------------------------------------------------------


async def _fetch_via_browser(
    config: ScraperConfig,
    browser_context: BrowserContext,
) -> list[VideoInfo]:
    """Scroll through a TikTok profile page and collect video URLs.

    Uses two complementary sub-strategies:
      1. **Network interception** — captures ``/api/post/item_list/`` JSON
         responses as the page scrolls, yielding structured video data.
      2. **DOM parsing** — queries the rendered page for ``<a>`` tags that
         link to ``/video/`` paths.

    Both sets are merged and deduplicated by video ID.

    Args:
        config: The scraper configuration.
        browser_context: An initialised Playwright browser context.

    Returns:
        A deduplicated, sorted list of :class:`VideoInfo` objects.

    Raises:
        CreatorNotFoundError: If the profile page indicates the user does
            not exist.
        BrowserError: If the browser encounters an unrecoverable error.
    """
    page = await browser_context.new_page()

    # Collected videos keyed by video_id to deduplicate.
    videos_by_id: dict[str, VideoInfo] = {}

    # Attach network interception *before* navigation.
    _attach_api_interceptor(page, videos_by_id)

    try:
        profile_url = f"https://www.tiktok.com/@{config.username}"
        logger.info("Browser fallback: navigating to %s", profile_url)

        response = await page.goto(profile_url, wait_until="domcontentloaded")
        if response is not None and response.status == 404:
            raise CreatorNotFoundError(config.username)

        await _wait_for_initial_content(page)

        # Dismiss cookie/consent banners and CAPTCHAs.
        await _dismiss_overlays(page)

        # Check for "user not found" indicators in the DOM.
        await _check_creator_exists(page, config.username)

        # ---- Initial hydration data extraction ----
        await _extract_hydration_data(page, videos_by_id)

        # ---- Scroll loop ----
        stale_count = 0
        previous_count = len(videos_by_id)

        while True:
            # Check limit early.
            if config.limit is not None and len(videos_by_id) >= config.limit:
                logger.info("Reached video limit (%d). Stopping scroll.", config.limit)
                break

            # Collect from DOM.
            await _collect_from_dom(page, videos_by_id)

            # Scroll to the bottom.
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await random_delay(config.scroll_delay_min, config.scroll_delay_max)

            current_count = len(videos_by_id)
            if current_count == previous_count:
                stale_count += 1
                logger.debug(
                    "No new videos after scroll (%d/%d stale).",
                    stale_count,
                    _MAX_STALE_SCROLLS,
                )
                if stale_count >= _MAX_STALE_SCROLLS:
                    logger.info(
                        "No new videos after %d scrolls. Assuming end of feed.",
                        _MAX_STALE_SCROLLS,
                    )
                    break
            else:
                stale_count = 0
                previous_count = current_count
                logger.info("Discovered %d videos so far...", current_count)

        # Final DOM collection after scrolling has finished.
        await _collect_from_dom(page, videos_by_id)

    except CreatorNotFoundError:
        raise
    except Exception as exc:
        raise BrowserError(f"Browser error while scraping: {exc}") from exc
    finally:
        await page.close()

    all_videos = sorted(videos_by_id.values(), key=lambda v: v.video_id)

    if config.limit is not None:
        all_videos = all_videos[: config.limit]

    logger.info(
        "Browser fallback: %d videos found for @%s.",
        len(all_videos),
        config.username,
    )
    return all_videos


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------


async def _wait_for_initial_content(page: Page) -> None:
    """Wait briefly for profile content or hydration data to become available."""
    try:
        await page.wait_for_function(
            """() => Boolean(
                document.querySelector('a[href*="/video/"], a[href*="/photo/"]') ||
                window.SIGI_STATE ||
                window.__UNIVERSAL_DATA_FOR_REHYDRATION__
            )""",
            timeout=5000,
        )
    except TimeoutError:
        logger.debug("Timed out waiting for initial profile content.")


async def _dismiss_overlays(page: Page) -> None:
    """Dismiss cookie consent banners, CAPTCHAs, and other overlays.

    TikTok often shows overlays that block page interaction. This function
    attempts to click through common dismiss/accept buttons via both
    Playwright selectors and direct JS clicks.
    """
    # Try JS-based CAPTCHA close first (bypasses overlay interception).
    captcha_closed: bool = await page.evaluate("""() => {
        const btn = document.getElementById('captcha_close_button');
        if (!btn) return false;
        btn.click();
        return true;
    }""")
    if captcha_closed:
        await page.wait_for_timeout(500)

    dismiss_selectors = [
        # Cookie consent banner buttons
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("Accept cookies")',
        'button:has-text("Allow all")',
        'button:has-text("Decline optional")',
        # CAPTCHA / age verification
        'button:has-text("Confirm")',
        # Generic close buttons on modals
        '[data-e2e="modal-close-inner-button"]',
        # Login prompt dismiss
        'div[role="dialog"] button[aria-label="Close"]',
    ]
    for selector in dismiss_selectors:
        try:
            button = await page.query_selector(selector)
            if button is not None and await button.is_visible():
                await button.click()
                logger.debug("Dismissed overlay: %s", selector)
                await page.wait_for_timeout(1000)
        except Exception:
            # Non-fatal — overlay dismissal is best-effort.
            pass


async def _check_creator_exists(page: Page, username: str) -> None:
    """Raise :class:`CreatorNotFoundError` if the profile page shows a 404.

    Uses specific data-e2e attributes and page title checks rather than
    broad text matching, to avoid false positives during page load.
    """
    # Check data-e2e attribute — TikTok's most reliable indicator.
    not_found_el = await page.query_selector('[data-e2e="user-not-found"]')
    if not_found_el is not None:
        raise CreatorNotFoundError(username)

    # Check the page title for 404 patterns.
    title = await page.title()
    title_lower = title.lower()
    if any(
        phrase in title_lower
        for phrase in ("not found", "couldn't find", "could not find", "404")
    ):
        raise CreatorNotFoundError(username)

    # Check for specific "not found" heading elements (more targeted than
    # bare text matching which can false-positive during loading).
    not_found_headings = await page.query_selector_all(
        'h1, h2, [class*="NotFound"], [class*="not-found"], p[class*="error"]'
    )
    for heading in not_found_headings:
        text = await heading.text_content()
        if text is not None:
            lower = text.lower().strip()
            if lower in (
                "couldn't find this account",
                "this account was banned",
                "couldn't find this page",
                "page not available",
            ):
                raise CreatorNotFoundError(username)


def _attach_api_interceptor(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
) -> None:
    """Listen for ``/api/post/item_list/`` responses and extract video data."""

    async def _on_response(response: Response) -> None:
        if _API_ITEM_LIST_FRAGMENT not in response.url:
            return
        try:
            body = await response.body()
            data: dict[str, Any] = json.loads(body)
            for item in data.get("itemList", []):
                vid = str(item.get("id", ""))
                author = item.get("author", {}).get("uniqueId", "")
                if vid:
                    url = f"https://www.tiktok.com/@{author}/video/{vid}"
                    if vid not in videos_by_id:
                        videos_by_id[vid] = VideoInfo(video_id=vid, url=url)
        except Exception:
            # Non-fatal — we have the DOM fallback.
            logger.debug("Failed to parse API response.", exc_info=True)

    page.on("response", _on_response)


async def _extract_hydration_data(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
) -> None:
    """Extract video data from TikTok's server-side hydration scripts."""
    try:
        data_json: str | None = await page.evaluate("""
            () => {
                // Try SIGI_STATE first (older format).
                if (window.SIGI_STATE) {
                    return JSON.stringify(window.SIGI_STATE);
                }
                // Then the newer __UNIVERSAL_DATA_FOR_REHYDRATION__.
                if (window.__UNIVERSAL_DATA_FOR_REHYDRATION__) {
                    return JSON.stringify(
                        window.__UNIVERSAL_DATA_FOR_REHYDRATION__
                    );
                }
                return null;
            }
        """)
        if data_json is None:
            return

        data: dict[str, Any] = json.loads(data_json)
        _parse_hydration_items(data, videos_by_id)

    except Exception:
        logger.debug("Hydration data extraction failed.", exc_info=True)


def _parse_hydration_items(
    data: dict[str, Any],
    videos_by_id: dict[str, VideoInfo],
) -> None:
    """Walk a hydration-state dict and pull out video items."""
    # SIGI_STATE format: ItemModule -> { videoId: {...} }
    item_module = data.get("ItemModule", {})
    if isinstance(item_module, dict):
        for vid, item_data in item_module.items():
            if isinstance(item_data, dict):
                author = item_data.get("author", "")
                url = f"https://www.tiktok.com/@{author}/video/{vid}"
                if vid not in videos_by_id:
                    videos_by_id[vid] = VideoInfo(video_id=vid, url=url)

    # __UNIVERSAL_DATA format: nested under __DEFAULT_SCOPE__
    default_scope = data.get("__DEFAULT_SCOPE__", {})
    user_detail = default_scope.get("webapp.user-detail", {})
    user_post = default_scope.get("webapp.user-post", {})

    # Try both potential locations for item lists.
    for section in (user_detail, user_post):
        if not isinstance(section, dict):
            continue
        item_list = section.get("itemList", [])
        if isinstance(item_list, list):
            for item in item_list:
                if isinstance(item, dict):
                    vid = str(item.get("id", ""))
                    author = item.get("author", {})
                    if isinstance(author, dict):
                        author = author.get("uniqueId", "")
                    if vid:
                        url = f"https://www.tiktok.com/@{author}/video/{vid}"
                        if vid not in videos_by_id:
                            videos_by_id[vid] = VideoInfo(video_id=vid, url=url)


async def _collect_from_dom(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
) -> None:
    """Query the DOM for video and photo links and add them to *videos_by_id*."""
    for selector in (_VIDEO_LINK_SELECTOR, _PHOTO_LINK_SELECTOR):
        links = await page.query_selector_all(selector)
        for link in links:
            href = await link.get_attribute("href")
            if href is None:
                continue
            vid = extract_post_id(href)
            if vid is None:
                continue
            # Normalise to a full URL.
            if href.startswith("/"):
                href = f"https://www.tiktok.com{href}"
            if vid not in videos_by_id:
                videos_by_id[vid] = VideoInfo(video_id=vid, url=href)
