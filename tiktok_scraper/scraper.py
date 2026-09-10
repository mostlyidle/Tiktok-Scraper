"""Core scraping logic — discover video URLs for a TikTok creator.

Primary strategy uses yt-dlp's ``TikTokUser`` extractor which handles
TikTok's anti-bot protections natively.  An optional Playwright-based
fallback (browser scrolling + DOM parsing + API interception) is
retained for situations where yt-dlp extraction fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import yt_dlp
from curl_cffi import requests as curl_requests
from playwright.async_api import BrowserContext, Page, Response, TimeoutError
from yt_dlp.networking.impersonate import ImpersonateTarget
from yt_dlp.utils import DownloadError as YtdlpDownloadError

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.exceptions import BrowserError, CreatorNotFoundError
from tiktok_scraper.models import VideoInfo
from tiktok_scraper.utils import (
    RequestLimiter,
    bounded_backoff,
    create_stealth_browser_context,
    get_downloaded_ids,
    isolated_cookie_file,
    parse_retry_after,
    random_delay,
    read_netscape_cookie_pairs,
    redact_log_value,
)

logger = logging.getLogger(__name__)


class _YtdlpLogger:
    """Route yt-dlp output through this module's logger."""

    def debug(self, message: str) -> None:
        logger.debug("yt-dlp: %s", _redact_log_value(message))

    def warning(self, message: str) -> None:
        logger.warning("yt-dlp: %s", _redact_log_value(message))

    def error(self, message: str) -> None:
        logger.error("yt-dlp: %s", _redact_log_value(message))


# Maximum consecutive scrolls with no new videos before we stop (browser fallback).
_MAX_STALE_SCROLLS = 5

# Selectors / URL fragments used to identify video links (browser fallback).
_VIDEO_LINK_SELECTOR = 'a[href*="/video/"]'
_PHOTO_LINK_SELECTOR = 'a[href*="/photo/"]'
_API_ITEM_LIST_FRAGMENTS = (
    "/api/post/item_list/",
    "/api/creator/item_list/",
)
_OFFICIAL_TIKTOK_HOSTS = frozenset(
    {
        "tiktok.com",
        "www.tiktok.com",
        "m.tiktok.com",
        "tiktokv.com",
        "www.tiktokv.com",
    }
)
_POST_ID_PATTERN = re.compile(r"^[0-9]{15,25}$")
_CHANNEL_ID_PATTERN = re.compile(r"^MS4wLjAB[A-Za-z0-9_-]{20,200}$")
_AUTHOR_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,24}$")
_POST_PATH_PATTERN = re.compile(
    r"^/@(?P<author>[A-Za-z0-9._-]{1,200})/"
    r"(?P<kind>video|photo)/(?P<id>[0-9]{15,25})/?$",
    re.IGNORECASE,
)
_FRONTITY_STATE_PATTERN = re.compile(
    r'<script[^>]+id=["\']__FRONTITY_CONNECT_STATE__["\'][^>]*>'
    r"(?P<data>.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)
_MAX_EMBED_POST_PROBES = 8
_PERMANENT_YTDLP_ERRORS = (
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
_IMPERSONATE_CHROME = ImpersonateTarget.from_str("chrome")
_YTDLP_LOGGER = _YtdlpLogger()
_AUTHOR_KEYS = (
    "author",
    "authorInfo",
    "authorInfos",
    "authorName",
    "author_name",
    "uniqueId",
    "unique_id",
    "userName",
    "username",
    "creator",
    "uploader",
)
_ID_KEYS = ("id", "video_id", "videoId", "post_id", "postId")
_URL_KEYS = (
    "webpage_url",
    "webpageUrl",
    "original_url",
    "originalUrl",
    "url",
)
_INCONCLUSIVE_STATUS_CODES = frozenset({401, 403, 429})
_BLOCKED_PAGE_PHRASES = (
    "account is private",
    "access denied",
    "are you a robot",
    "complete the security check",
    "internal server error",
    "captcha",
    "challenge required",
    "log in to continue",
    "log in to view",
    "login to continue",
    "login to view",
    "private account",
    "press and hold",
    "profile is private",
    "request blocked",
    "security check",
    "sign in to continue",
    "sign in to view",
    "service unavailable",
    "temporarily unavailable",
    "temporarily blocked",
    "this account is private",
    "too many requests",
    "unusual traffic",
    "verify you are human",
    "verify that you're human",
    "verify to continue",
    "videos are private",
    "you have been blocked",
    "you must log in",
)
_BLOCKED_TITLE_PHRASES = (
    "account is private",
    "access denied",
    "captcha verification",
    "challenge required",
    "human verification",
    "private account",
    "security check",
    "temporarily unavailable",
    "too many requests",
    "log in to continue",
    "log in to view",
    "login to continue",
    "login to view",
    "sign in to continue",
    "sign in to view",
    "verify you",
    "verification required",
)
_BLOCKED_ATTRIBUTE_PATTERN = re.compile(
    r"(?:captcha[-_](?:verify|challenge|modal)|"
    r"challenge[-_](?:verify|required|modal)|"
    r"security[-_](?:check|verify)|"
    r"login[-_](?:modal|wall)|"
    r"(?:private|access)[-_](?:account|denied)|"
    r"verify[-_](?:human|robot))",
    re.IGNORECASE,
)
def _redact_log_value(value: object) -> str:
    """Remove query strings and common credential values from log messages."""
    return redact_log_value(value)


def _valid_post_id(value: Any) -> str | None:
    """Return a post ID only when it is a plain 15-25 digit value."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, str):
        candidate = value
    else:
        return None
    return candidate if _POST_ID_PATTERN.fullmatch(candidate) else None


def _valid_author(value: Any) -> str | None:
    """Return a syntactically valid TikTok handle, without an at-sign."""
    if not isinstance(value, str):
        return None
    author = value.strip()
    if author.startswith("@"):
        author = author[1:]
    if author in {".", ".."} or not _AUTHOR_PATTERN.fullmatch(author):
        return None
    return author


def _is_opaque_channel_author(value: str) -> bool:
    """Return whether a URL author is yt-dlp's opaque channel identifier."""
    return _CHANNEL_ID_PATTERN.fullmatch(value) is not None


def _author_from_value(value: Any) -> str | None:
    """Read a unique creator handle from a TikTok author-shaped value."""
    if isinstance(value, str):
        return _valid_author(value)
    if not isinstance(value, dict):
        return None

    for key in (
        "uniqueId",
        "unique_id",
        "username",
        "userName",
        "user_name",
        "authorName",
        "author_name",
    ):
        if key in value:
            return _valid_author(value[key])

    for key in ("author", "authorInfo", "authorInfos", "user"):
        if key in value:
            author = _author_from_value(value[key])
            if author is not None:
                return author
    return None


def _authors_from_value(value: Any) -> tuple[set[str], bool]:
    """Collect all username aliases and report malformed/conflicting data."""
    if isinstance(value, str):
        author = _valid_author(value)
        return ({author.casefold()} if author is not None else set(), author is None)
    if not isinstance(value, dict):
        return set(), True

    authors: set[str] = set()
    invalid = False
    value_keys = (
        "uniqueId",
        "unique_id",
        "username",
        "userName",
        "user_name",
        "authorName",
        "author_name",
    )
    nested_keys = ("author", "authorInfo", "authorInfos", "user")
    seen_alias = False
    for key in value_keys:
        if key not in value:
            continue
        seen_alias = True
        author = _valid_author(value[key])
        if author is None:
            invalid = True
        else:
            authors.add(author.casefold())
    for key in nested_keys:
        if key not in value:
            continue
        seen_alias = True
        nested_authors, nested_invalid = _authors_from_value(value[key])
        authors.update(nested_authors)
        invalid = invalid or nested_invalid
    return authors, invalid or (seen_alias and not authors)


def _record_authors(record: dict[str, Any]) -> tuple[bool, set[str], bool]:
    """Return whether author data exists, its values, and invalid-data state."""
    present = False
    authors: set[str] = set()
    invalid = False
    for key in _AUTHOR_KEYS:
        if key not in record:
            continue
        present = True
        record_authors, record_invalid = _authors_from_value(record[key])
        authors.update(record_authors)
        invalid = invalid or record_invalid
    return present, authors, invalid


def _first_record_author(record: dict[str, Any]) -> str | None:
    """Return the first valid author value in source spelling."""
    for key in _AUTHOR_KEYS:
        if key in record:
            author = _author_from_value(record[key])
            if author is not None:
                return author
    return None


def _record_post_id(record: dict[str, Any]) -> str | None:
    """Read a consistent valid post ID from all ID aliases in a record."""
    values: list[str] = []
    for key in _ID_KEYS:
        if key not in record:
            continue
        post_id = _valid_post_id(record[key])
        if post_id is None:
            return None
        values.append(post_id)
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def _record_is_photo(record: dict[str, Any]) -> bool:
    """Return whether a source record explicitly describes a photo post."""
    for key in (
        "imagePost",
        "image_post",
        "imagePostInfo",
        "image_post_info",
        "images",
        "imageList",
        "image_list",
        "displayImages",
        "display_images",
    ):
        value = record.get(key)
        if isinstance(value, (dict, list)) and value:
            return True
    for key in ("itemType", "item_type", "postType", "post_type", "type"):
        value = record.get(key)
        if isinstance(value, str) and value.casefold() in {
            "photo",
            "image",
            "imagepost",
            "photo_post",
        }:
            return True
    return False


def _canonical_post_url(
    url: Any,
    requested_username: str | None = None,
) -> tuple[str, str, str] | None:
    """Parse an official TikTok post URL and return canonical parts.

    The returned tuple is ``(post_id, post_kind, author)``. Query strings and
    fragments are intentionally discarded from the canonical URL.
    """
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None

    try:
        parsed = urlsplit(value)
        host = parsed.hostname.casefold() if parsed.hostname else None
        if parsed.netloc:
            if (
                parsed.scheme.casefold() not in {"", "https"}
                or host not in _OFFICIAL_TIKTOK_HOSTS
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port is not None
            ):
                return None
        elif not value.startswith("/"):
            return None
        path = unquote(parsed.path)
    except (TypeError, ValueError):
        return None

    match = _POST_PATH_PATTERN.fullmatch(path)
    if match is None:
        return None
    raw_author = match.group("author")
    author = _valid_author(raw_author)
    if author is None and not _is_opaque_channel_author(raw_author):
        return None
    post_id = match.group("id")
    kind = match.group("kind").casefold()
    if (
        requested_username is not None
        and not _is_opaque_channel_author(raw_author)
        and raw_author.casefold() != requested_username.casefold()
    ):
        return None
    # Preserve opaque channel IDs so callers can explicitly opt into the
    # channel-route format instead of silently treating one as a username.
    return post_id, kind, author or raw_author


def _canonical_profile_author(url: Any) -> str | None:
    """Read the author from an official TikTok profile URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    value = url.strip()
    try:
        parsed = urlsplit(value)
        host = parsed.hostname.casefold() if parsed.hostname else None
        if parsed.netloc and (
            parsed.scheme.casefold() not in {"", "https"}
            or host not in _OFFICIAL_TIKTOK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            return None
        if parsed.netloc == "" and not value.startswith("/"):
            return None
        path = unquote(parsed.path)
    except (TypeError, ValueError):
        return None
    match = re.fullmatch(r"/@(?P<author>[A-Za-z0-9._]{1,24})/?", path)
    return _valid_author(match.group("author")) if match is not None else None


def _record_to_video_info(
    record: dict[str, Any],
    requested_username: str | None,
    *,
    allow_missing_url: bool,
    allow_opaque_channel_author: bool = False,
    expected_opaque_channel_author: str | None = None,
    default_kind: str = "video",
) -> VideoInfo | None:
    """Validate one source record and canonicalize it to a TikTok URL."""
    if not isinstance(record, dict):
        return None

    id_present = any(key in record for key in _ID_KEYS)
    post_id = _record_post_id(record) if id_present else None
    if id_present and post_id is None:
        return None

    _present, authors, invalid_author = _record_authors(record)
    if invalid_author or len(authors) > 1:
        return None

    url_parts: list[tuple[str, str, str]] = []
    for key in _URL_KEYS:
        if key not in record or record[key] is None or record[key] == "":
            continue
        parsed_url = _canonical_post_url(record[key], requested_username)
        if parsed_url is None:
            return None
        url_parts.append(parsed_url)

    if url_parts:
        first_parts = url_parts[0]
        if any(parts != first_parts for parts in url_parts[1:]):
            return None
        url_id, kind, url_author = first_parts
        if post_id is not None and post_id != url_id:
            return None
        post_id = url_id
        if _is_opaque_channel_author(url_author):
            if (
                not allow_opaque_channel_author
                or expected_opaque_channel_author is None
                or url_author != expected_opaque_channel_author
            ):
                return None
        else:
            authors.add(url_author.casefold())
    elif not allow_missing_url:
        return None

    if post_id is None or not _valid_post_id(post_id):
        return None
    if not authors and not (
        allow_opaque_channel_author
        and requested_username is not None
        and url_parts
        and all(_is_opaque_channel_author(parts[2]) for parts in url_parts)
    ):
        # An ID without a URL or author cannot be attributed safely.
        return None
    canonical_author: str
    if requested_username is not None:
        requested_key = requested_username.casefold()
        if any(author != requested_key for author in authors):
            return None
        canonical_author = requested_username
    else:
        if len(authors) != 1:
            return None
        if url_parts:
            canonical_author = url_parts[0][2]
        else:
            record_author = _first_record_author(record)
            if record_author is None:
                return None
            canonical_author = record_author

    if url_parts:
        kind = "photo" if _record_is_photo(record) else url_parts[0][1]
    elif default_kind not in {"video", "photo"}:
        return None
    else:
        kind = "photo" if _record_is_photo(record) else default_kind
    return VideoInfo(
        video_id=post_id,
        url=f"https://www.tiktok.com/@{canonical_author}/{kind}/{post_id}",
    )


def _add_record(
    videos_by_id: dict[str, VideoInfo],
    record: dict[str, Any],
    requested_username: str | None,
    *,
    allow_missing_url: bool,
    allow_opaque_channel_author: bool = False,
    expected_opaque_channel_author: str | None = None,
    default_kind: str = "video",
) -> bool:
    """Validate and add a source record while retaining first-seen order."""
    video = _record_to_video_info(
        record,
        requested_username,
        allow_missing_url=allow_missing_url,
        allow_opaque_channel_author=allow_opaque_channel_author,
        expected_opaque_channel_author=expected_opaque_channel_author,
        default_kind=default_kind,
    )
    if video is None or video.video_id in videos_by_id:
        return False
    videos_by_id[video.video_id] = video
    return True


def _sanitize_video_infos(
    videos: Iterable[Any],
    requested_username: str,
    limit: int | None,
) -> list[VideoInfo]:
    """Validate strategy output before exposing it through the public API."""
    if limit == 0:
        return []
    videos_by_id: dict[str, VideoInfo] = {}
    for video in videos:
        if not isinstance(video, VideoInfo):
            continue
        _add_record(
            videos_by_id,
            {"id": video.video_id, "url": video.url},
            requested_username,
            allow_missing_url=False,
        )
    result = list(videos_by_id.values())
    return result[:limit] if limit is not None else result


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
        A deduplicated, feed-ordered list of :class:`VideoInfo` objects.

    Raises:
        CreatorNotFoundError: If the profile does not exist.
        BrowserError: If both strategies fail.
    """
    # A zero limit is a complete request, not a request for an empty feed.
    # Return before either strategy so no network or browser setup occurs.
    if config.limit == 0:
        logger.info("Video limit is 0; skipping discovery.")
        return []

    # ---- Strategy 1: yt-dlp (primary) ----
    try:
        videos = _sanitize_video_infos(
            await _fetch_via_ytdlp(config),
            config.username,
            config.limit,
        )
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
            _redact_log_value(exc),
        )

    # ---- Strategy 2: Playwright browser (fallback) ----
    if browser_context is not None:
        return _sanitize_video_infos(
            await _fetch_via_browser(config, browser_context),
            config.username,
            config.limit,
        )

    logger.info("Attempting Playwright browser fallback...")
    try:
        pw, browser, context = await create_stealth_browser_context(
            headless=config.headless,
            cookies_file=config.cookies_file,
        )
    except Exception as exc:
        raise BrowserError(
            f"Could not start browser fallback: {_redact_log_value(exc)}"
        ) from exc

    primary_error: BaseException | None = None
    try:
        return _sanitize_video_infos(
            await _fetch_via_browser(config, context),
            config.username,
            config.limit,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        await _cleanup_browser_resources(
            pw,
            browser,
            context,
            primary_error,
        )


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
    if config.limit == 0:
        return []

    profile_url = f"https://www.tiktok.com/@{config.username}"
    logger.info("Discovering videos via yt-dlp for %s", profile_url)

    ydl_opts = _build_discovery_opts(config)

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
        logger.warning(
            "yt-dlp profile discovery failed for @%s: %s. "
            "Trying embed/channel fallback.",
            config.username,
            _redact_log_value(exc),
        )
        try:
            fallback_videos = await _fetch_via_embed_channel(config)
        except Exception as fallback_exc:
            logger.debug(
                "Embed/channel fallback failed: %s",
                _redact_log_value(fallback_exc),
            )
            raise exc from fallback_exc
        if fallback_videos:
            return _sanitize_video_infos(
                fallback_videos,
                config.username,
                config.limit,
            )
        raise

    if info is None:
        return []

    if _info_indicates_inconclusive(info):
        raise BrowserError(
            "yt-dlp received a blocked or unusable profile response; "
            "discovery is inconclusive."
        )
    if _info_has_wrong_creator(info, config.username):
        raise BrowserError(
            "yt-dlp returned a different creator profile; discovery is inconclusive."
        )

    all_videos = _video_infos_from_info(info, config.limit, config.username)

    logger.info(
        "yt-dlp discovered %d videos for @%s.",
        len(all_videos),
        config.username,
    )
    return all_videos


async def _fetch_via_ytdlp_channel(
    config: ScraperConfig,
    channel_id: str,
) -> list[VideoInfo]:
    """Discover posts through yt-dlp's stable secondary-user-ID route."""
    if config.limit == 0:
        return []

    channel_url = f"tiktokuser:{channel_id}"
    logger.info("Discovering videos via TikTok channel ID for @%s", config.username)
    ydl_opts = _build_discovery_opts(config)
    loop = asyncio.get_running_loop()

    async def _extract() -> dict[str, Any] | None:
        return await loop.run_in_executor(
            None,
            functools.partial(_sync_extract, channel_url, ydl_opts),
        )

    info = await _retry_ytdlp_operation(_extract, config, "channel discovery")
    if info is None:
        return []
    if _info_indicates_inconclusive(info):
        raise BrowserError(
            "yt-dlp received a blocked or unusable channel response; "
            "discovery is inconclusive."
        )
    if _info_has_wrong_creator(info, config.username):
        raise BrowserError(
            "yt-dlp returned a different creator profile; discovery is inconclusive."
        )

    videos = _video_infos_from_info(
        info,
        config.limit,
        config.username,
        allow_opaque_channel_author=True,
        expected_opaque_channel_author=channel_id,
    )
    logger.info(
        "Channel-ID discovery found %d videos for @%s.",
        len(videos),
        config.username,
    )
    return videos


def _build_discovery_opts(config: ScraperConfig) -> dict[str, Any]:
    """Construct yt-dlp options shared by profile and channel discovery."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": False,
        "logger": _YTDLP_LOGGER,
        "extract_flat": True,
        "sleep_interval_requests": config.request_delay,
        "sleep_interval": config.request_delay,
        "max_sleep_interval": 2 * config.request_delay,
        "socket_timeout": config.request_timeout,
        "impersonate": _IMPERSONATE_CHROME,
        # yt-dlp needs a small internal challenge retry to hydrate profile
        # pages; the surrounding loop still owns application-level retries.
        "extractor_retries": max(1, min(config.download_retries, 2)),
    }
    if config.limit is not None:
        opts["playlistend"] = config.limit
    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)
    return opts


def _video_infos_from_info(
    info: dict[str, Any],
    limit: int | None,
    requested_username: str | None = None,
    *,
    allow_opaque_channel_author: bool = False,
    expected_opaque_channel_author: str | None = None,
) -> list[VideoInfo]:
    """Convert a yt-dlp playlist result into validated, ordered records."""
    if limit == 0 or not isinstance(info, dict):
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
        _add_record(
            videos_by_id,
            entry,
            requested_username,
            allow_missing_url=True,
            allow_opaque_channel_author=allow_opaque_channel_author,
            expected_opaque_channel_author=expected_opaque_channel_author,
        )

    all_videos = list(videos_by_id.values())
    if limit is not None:
        all_videos = all_videos[:limit]
    return all_videos


def _info_indicates_inconclusive(info: dict[str, Any]) -> bool:
    """Recognize blocked or gated states in a yt-dlp profile payload."""
    if _hydration_indicates_inconclusive(info):
        return True
    for key in ("http_status", "status_code", "statusCode", "status"):
        status = info.get(key)
        if isinstance(status, int) and _status_is_inconclusive(status):
            return True
        if (
            isinstance(status, str)
            and status.isdigit()
            and _status_is_inconclusive(int(status))
        ):
            return True
        if isinstance(status, str) and any(
            phrase in status.casefold()
            for phrase in ("blocked", "challenge", "login", "private")
        ):
            return True

    for key in (
        "challenge",
        "is_private",
        "isPrivate",
        "login_required",
        "loginRequired",
        "private",
        "requires_login",
        "requiresLogin",
    ):
        if info.get(key) is True:
            return True

    for key in ("availability", "error", "title"):
        value = info.get(key)
        if not isinstance(value, str):
            continue
        value_lower = value.casefold()
        if _html_indicates_inconclusive(value):
            return True
        if key in {"availability", "error"} and any(
            phrase in value_lower
            for phrase in (
                "blocked",
                "captcha",
                "challenge",
                "login required",
                "private",
                "requires login",
            )
        ):
            return True
    return False


def _info_has_wrong_creator(info: dict[str, Any], username: str) -> bool:
    """Detect explicit creator metadata in a playlist for another handle."""
    _present, authors, invalid = _record_authors(info)
    requested_key = username.casefold()
    if invalid and any(key in info for key in _AUTHOR_KEYS):
        return True
    if any(author != requested_key for author in authors):
        return True

    for key in ("webpage_url", "webpageUrl", "original_url", "originalUrl", "url"):
        value = info.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        post = _canonical_post_url(value)
        if post is not None and post[2].casefold() != requested_key:
            return True
        profile_author = _canonical_profile_author(value)
        if profile_author is not None and profile_author.casefold() != requested_key:
            return True
        if (
            post is None
            and profile_author is None
            and value.casefold().startswith(("http://", "https://", "//"))
        ):
            return True
    return False


def _hydration_indicates_inconclusive(value: Any) -> bool:
    """Find blocked/gated response metadata anywhere in hydration JSON."""
    if isinstance(value, dict):
        for key in ("http_status", "status_code", "statusCode", "status"):
            status = value.get(key)
            if isinstance(status, int) and _status_is_inconclusive(status):
                return True
            if (
                isinstance(status, str)
                and status.isdigit()
                and _status_is_inconclusive(int(status))
            ):
                return True
            if isinstance(status, str) and any(
                phrase in status.casefold()
                for phrase in (
                    "blocked",
                    "captcha",
                    "challenge",
                    "forbidden",
                    "login",
                    "private",
                    "too many requests",
                )
            ):
                return True
        for key in (
            "challenge",
            "is_private",
            "isPrivate",
            "login_required",
            "loginRequired",
            "private",
            "requires_login",
            "requiresLogin",
        ):
            if value.get(key) is True:
                return True
        for key in ("error", "status_msg", "statusMsg", "message", "msg"):
            message = value.get(key)
            if isinstance(message, str) and any(
                phrase in message.casefold()
                for phrase in (
                    "blocked",
                    "captcha",
                    "challenge",
                    "forbidden",
                    "login",
                    "private",
                    "too many requests",
                )
            ):
                return True
        return any(_hydration_indicates_inconclusive(child) for child in value.values())
    if isinstance(value, list):
        return any(_hydration_indicates_inconclusive(child) for child in value)
    return False


def _sync_extract(url: str, opts: dict[str, Any]) -> dict[str, Any] | None:
    """Synchronous yt-dlp extraction (runs inside an executor)."""
    with _isolated_ydl_opts(opts) as safe_opts, yt_dlp.YoutubeDL(safe_opts) as ydl:
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


@contextlib.contextmanager
def _isolated_ydl_opts(opts: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Use a private cookie-file copy for one yt-dlp invocation."""
    cookiefile = opts.get("cookiefile")
    if not isinstance(cookiefile, str):
        yield opts
        return
    with isolated_cookie_file(Path(cookiefile)) as isolated:
        if isolated is None or str(isolated) == cookiefile:
            yield opts
        else:
            safe_opts = dict(opts)
            safe_opts["cookiefile"] = str(isolated)
            yield safe_opts


async def _fetch_via_embed_channel(config: ScraperConfig) -> list[VideoInfo]:
    """Discover posts through the profile embed's secondary user ID.

    The normal TikTok profile extractor needs a post page to resolve the
    account's ``secUid``.  TikTok's embed pages expose that value separately,
    so this path can recover when the normal profile pages are WAF-challenged.
    """
    if config.limit == 0:
        return []

    loop = asyncio.get_running_loop()
    request_limiter = RequestLimiter(config.request_delay)
    profile_url = f"https://www.tiktok.com/embed/@{config.username}"
    post_ids: list[str] = []

    try:
        profile_html = await _fetch_embed_html_with_retry(
            profile_url,
            config,
            loop,
            request_limiter,
        )
        if _html_indicates_inconclusive(profile_html):
            raise RuntimeError("TikTok profile embed was blocked or challenged")
        profile_ids = _extract_embed_profile_post_ids(profile_html, config.username)
        post_ids.extend(profile_ids)
    except Exception as exc:
        logger.debug(
            "TikTok profile embed discovery failed: %s",
            _redact_log_value(exc),
        )

    # A downloaded ID is only a probe candidate. It is not trusted as evidence
    # that the post belongs to this creator; the post embed must identify the
    # requested author below before its channel ID is used.
    try:
        cached_ids = {
            post_id
            for post_id in get_downloaded_ids(config.output_dir)
            if _valid_post_id(post_id) is not None
        }
    except Exception as exc:
        logger.debug("Could not inspect cached post IDs: %s", _redact_log_value(exc))
        cached_ids = set()
    post_ids.extend(sorted(cached_ids, reverse=True))
    post_ids = list(dict.fromkeys(post_ids))[:_MAX_EMBED_POST_PROBES]
    if not post_ids:
        logger.warning(
            "TikTok embed returned no candidate posts for @%s.", config.username
        )
        return []

    channel_id: str | None = None
    for attempt, post_id in enumerate(post_ids):
        if attempt and config.request_delay:
            await asyncio.sleep(config.request_delay)
        video_url = f"https://www.tiktok.com/embed/v2/{post_id}"
        try:
            video_html = await _fetch_embed_html_with_retry(
                video_url,
                config,
                loop,
                request_limiter,
            )
            author = _extract_embed_author(video_html, post_id)
            if author is None:
                logger.debug(
                    "Ignoring embed probe %s because its author could not be "
                    "verified as @%s.",
                    post_id,
                    config.username,
                )
                continue
            if author.casefold() != config.username.casefold():
                logger.debug(
                    "Ignoring embed probe %s because its author was not "
                    "verified as @%s.",
                    post_id,
                    config.username,
                )
                continue
            channel_id = _extract_embed_channel_id(video_html, post_id)
        except Exception as exc:
            logger.debug(
                "TikTok post embed lookup failed for %s: %s",
                post_id,
                _redact_log_value(exc),
            )
        if channel_id:
            break

    if channel_id is None:
        logger.warning(
            "Could not resolve a TikTok channel ID for @%s.", config.username
        )
        return []

    logger.info("Resolved TikTok channel ID for @%s via embed data.", config.username)
    return _sanitize_video_infos(
        await _fetch_via_ytdlp_channel(config, channel_id),
        config.username,
        config.limit,
    )


async def _fetch_embed_html_with_retry(
    url: str,
    config: ScraperConfig,
    loop: asyncio.AbstractEventLoop,
    request_limiter: RequestLimiter,
) -> str:
    """Fetch an embed page with bounded retry and shared request pacing."""
    attempts = max(1, config.download_retries)
    for attempt in range(1, attempts + 1):
        await request_limiter.acquire()
        try:
            return await loop.run_in_executor(
                None,
                functools.partial(
                    _sync_fetch_embed_html,
                    url,
                    config.cookies_file,
                    config.request_timeout,
                ),
            )
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_embed_error(exc):
                raise
            delay = bounded_backoff(
                attempt,
                base=config.retry_backoff_base,
                cap=config.retry_backoff_max,
                retry_after=_retry_after_from_exception(exc),
            )
            logger.warning(
                "Embed request failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                attempts,
                _redact_log_value(exc),
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("embed retry loop must return or raise")


def _is_retryable_embed_error(exc: Exception) -> bool:
    """Classify curl HTTP/network failures for embed retries."""
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    if isinstance(status, int):
        return status in {408, 425, 429} or status >= 500
    message = str(exc).casefold()
    return any(
        phrase in message
        for phrase in (
            "408",
            "425",
            "429",
            "rate limit",
            "too many requests",
            "timeout",
            "timed out",
            "connection",
            "network",
            "502",
            "503",
            "504",
            "server error",
        )
    )


def _is_official_https_url(value: Any) -> bool:
    """Return whether a URL can safely be used for TikTok page requests."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname.casefold() if parsed.hostname else None
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme.casefold() == "https"
        and hostname in _OFFICIAL_TIKTOK_HOSTS
        and parsed.username is None
        and parsed.password is None
        and port is None
    )


def _sync_fetch_embed_html(
    url: str,
    cookies_file: Path | None,
    timeout: float = 30.0,
) -> str:
    """Fetch a TikTok embed page using Chrome TLS impersonation."""
    current_url = url
    for _ in range(6):
        if not _is_official_https_url(current_url):
            raise RuntimeError("TikTok embed redirected to an untrusted URL")
        response = curl_requests.get(
            current_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.tiktok.com/",
            },
            cookies=read_netscape_cookie_pairs(cookies_file, url=current_url),
            impersonate="chrome",
            timeout=timeout,
            allow_redirects=False,
        )
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not isinstance(location, str) or not location:
                raise RuntimeError("TikTok embed redirect had no location")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()  # type: ignore[no-untyped-call]
        final_url = getattr(response, "url", current_url)
        if not isinstance(final_url, str):
            final_url = current_url
        if not _is_official_https_url(final_url):
            raise RuntimeError("TikTok embed returned an untrusted URL")
        return response.text
    raise RuntimeError("TikTok embed exceeded the redirect limit")


def _parse_frontity_state(html: str) -> dict[str, Any] | None:
    """Parse TikTok's Frontity hydration state from an embed page."""
    match = _FRONTITY_STATE_PATTERN.search(html)
    if match is None:
        return None
    try:
        state = json.loads(match.group("data"))
    except json.JSONDecodeError:
        return None
    return state if isinstance(state, dict) else None


def _frontity_route_data(
    state: dict[str, Any],
    route: str,
) -> Any:
    """Return a Frontity route payload, tolerating route-key casing."""
    source = state.get("source")
    data = source.get("data") if isinstance(source, dict) else None
    if not isinstance(data, dict):
        return None
    value = data.get(route)
    if value is not None:
        return value
    route_key = route.casefold()
    for key, candidate in data.items():
        if isinstance(key, str) and key.casefold() == route_key:
            return candidate
    return None


def _extract_embed_profile_post_ids(html: str, username: str) -> list[str]:
    """Extract ordered post IDs from a profile embed response."""
    ids: list[str] = []
    seen: set[str] = set()
    state = _parse_frontity_state(html)
    if state is not None:
        profile = _frontity_route_data(state, f"/embed/@{username}")
        video_list = profile.get("videoList", []) if isinstance(profile, dict) else []
        if isinstance(video_list, list):
            for item in video_list:
                if not isinstance(item, dict):
                    continue
                _present, authors, invalid = _record_authors(item)
                if invalid or any(author != username.casefold() for author in authors):
                    continue
                item_urls = [
                    item[key]
                    for key in _URL_KEYS
                    if key in item and item[key] is not None and item[key] != ""
                ]
                post_id: str | None = None
                if item_urls:
                    item_info = _record_to_video_info(
                        item,
                        username,
                        allow_missing_url=True,
                    )
                    if item_info is None:
                        continue
                    post_id = item_info.video_id
                else:
                    post_id = _record_post_id(item)
                if post_id is not None and post_id not in seen:
                    seen.add(post_id)
                    ids.append(post_id)

    href_pattern = re.compile(
        r"(?:href|content)\s*=\s*([\"'])(.*?)\1",
        re.IGNORECASE | re.DOTALL,
    )
    for match in href_pattern.finditer(html):
        post = _canonical_post_url(match.group(2), username)
        if post is None:
            continue
        post_id = post[0]
        if post_id not in seen:
            seen.add(post_id)
            ids.append(post_id)

    # Some embed responses render post paths outside anchor elements. Only
    # accept an explicit official host or a standalone relative path; the
    # boundary prevents extracting the path portion of an external URL.
    raw_link_pattern = re.compile(
        r"(?<![\w./=?&#-])(?:(?:https?:)?//(?:www\.|m\.)?tiktok\.com)?"
        r"/@[A-Za-z0-9._]{1,24}/(?:video|photo)/[0-9]{15,25}/?",
        re.IGNORECASE,
    )
    for match in raw_link_pattern.finditer(html):
        post = _canonical_post_url(match.group(0), username)
        if post is None:
            continue
        post_id = post[0]
        if post_id not in seen:
            seen.add(post_id)
            ids.append(post_id)
    return ids


def _extract_embed_channel_id(html: str, video_id: str) -> str | None:
    """Extract a TikTok ``secUid`` from a video embed response."""
    if _valid_post_id(video_id) is None:
        return None
    state = _parse_frontity_state(html)
    if state is None:
        return None
    video = _frontity_route_data(state, f"/embed/v2/{video_id}")
    if not isinstance(video, dict):
        return None
    video_data = video.get("videoData")
    if not isinstance(video_data, dict):
        return None
    channel_ids: set[str] = set()
    invalid_channel_id = False
    for key in ("authorInfos", "authorInfo", "author"):
        if key not in video_data:
            continue
        found_ids, invalid = _channel_ids_from_author(video_data[key])
        channel_ids.update(found_ids)
        invalid_channel_id = invalid_channel_id or invalid
    if invalid_channel_id or len(channel_ids) != 1:
        return None
    return next(iter(channel_ids))


def _extract_embed_author(html: str, video_id: str) -> str | None:
    """Extract an explicit creator handle from a post embed response."""
    if _valid_post_id(video_id) is None:
        return None
    state = _parse_frontity_state(html)
    if state is not None:
        video = _frontity_route_data(state, f"/embed/v2/{video_id}")
        if isinstance(video, dict):
            video_data = video.get("videoData")
            if isinstance(video_data, dict):
                authors: set[str] = set()
                invalid_author = False
                for key in ("authorInfos", "authorInfo", "author"):
                    if key not in video_data:
                        continue
                    found_authors, invalid = _authors_from_value(
                        video_data[key]
                    )
                    authors.update(found_authors)
                    invalid_author = invalid_author or invalid
                if not invalid_author and len(authors) == 1:
                    return next(iter(authors))

    href_pattern = re.compile(
        r"(?:href|content)\s*=\s*([\"'])(.*?)\1",
        re.IGNORECASE | re.DOTALL,
    )
    href_authors: set[str] = set()
    for match in href_pattern.finditer(html):
        post = _canonical_post_url(match.group(2))
        if post is not None and post[0] == video_id:
            href_authors.add(post[2].casefold())
    return next(iter(href_authors)) if len(href_authors) == 1 else None


def _find_nested_channel_id(value: Any) -> str | None:
    """Find a valid secondary user ID in an author-shaped value."""
    return _channel_id_from_author(value)


def _channel_ids_from_author(value: Any) -> tuple[set[str], bool]:
    """Collect every secondary ID from an author-shaped value."""
    if isinstance(value, dict):
        channel_ids: set[str] = set()
        invalid = False
        for key in ("secUid", "sec_uid", "channel_id", "channelId"):
            if key not in value:
                continue
            candidate = value[key]
            if isinstance(candidate, str) and _CHANNEL_ID_PATTERN.fullmatch(candidate):
                channel_ids.add(candidate)
            else:
                invalid = True
        for key in ("authorInfos", "authorInfo", "author", "user"):
            if key not in value:
                continue
            nested_ids, nested_invalid = _channel_ids_from_author(value[key])
            channel_ids.update(nested_ids)
            invalid = invalid or nested_invalid
        return channel_ids, invalid
    if isinstance(value, list):
        list_channel_ids: set[str] = set()
        invalid = False
        for child in value:
            nested_ids, nested_invalid = _channel_ids_from_author(child)
            list_channel_ids.update(nested_ids)
            invalid = invalid or nested_invalid
        return list_channel_ids, invalid
    return set(), False


def _channel_id_from_author(value: Any) -> str | None:
    """Read a secondary user ID without searching unrelated post fields."""
    channel_ids, invalid = _channel_ids_from_author(value)
    if invalid or len(channel_ids) != 1:
        return None
    return next(iter(channel_ids))


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
                _redact_log_value(exc),
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("retry loop must return or raise")


def _is_creator_not_found(message: str) -> bool:
    """Recognise creator-not-found errors without broad transient matches."""
    message = message.casefold()
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
    if _is_creator_not_found(message):
        return False
    if re.search(r"\b(?:http\s+error|status(?:\s+code)?)\s*:?[\s-]*5\d{2}\b", message):
        return True
    if re.search(
        r"\b(?:http\s+error|status(?:\s+code)?)\s*:?[\s-]*(?:401|403|404)\b", message
    ):
        return False
    if any(phrase in message for phrase in _PERMANENT_YTDLP_ERRORS):
        return False
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    return any(phrase in message for phrase in _TRANSIENT_YTDLP_ERRORS)


def _retry_after_from_message(message: str) -> float | None:
    """Extract a seconds hint from yt-dlp's human-readable error text."""
    match = re.search(r"retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)", message)
    return parse_retry_after(match.group(1)) if match else None


def _retry_after_from_exception(exc: Exception) -> float | None:
    """Read a Retry-After hint from an HTTP-like exception."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return _retry_after_from_message(str(exc).casefold())
    value = headers.get("Retry-After") or headers.get("retry-after")
    return parse_retry_after(str(value)) if value is not None else None


# ---------------------------------------------------------------------------
# Strategy 2 — Playwright browser (fallback)
# ---------------------------------------------------------------------------


async def _fetch_via_browser(
    config: ScraperConfig,
    browser_context: BrowserContext,
) -> list[VideoInfo]:
    """Scroll through a TikTok profile page and collect video URLs.

    Uses two complementary sub-strategies:
      1. **Network interception** — captures TikTok creator/post item-list JSON
         responses as the page scrolls, yielding structured video data.
      2. **DOM parsing** — queries the rendered page for ``<a>`` tags that
         link to ``/video/`` paths.

    Both sets are merged and deduplicated by video ID.

    Args:
        config: The scraper configuration.
        browser_context: An initialised Playwright browser context.

    Returns:
        A deduplicated, feed-ordered list of :class:`VideoInfo` objects.

    Raises:
        CreatorNotFoundError: If the profile page indicates the user does
            not exist.
        BrowserError: If the browser encounters an unrecoverable error.
    """
    if config.limit == 0:
        return []

    try:
        page = await browser_context.new_page()
    except Exception as exc:
        raise BrowserError(
            f"Could not create browser page: {_redact_log_value(exc)}"
        ) from exc

    # Dict insertion order is the feed/source order. It also gives us stable
    # first-seen deduplication when API, hydration, and DOM data overlap.
    videos_by_id: dict[str, VideoInfo] = {}
    pending_response_tasks: set[asyncio.Task[Any]] = set()
    response_state: dict[str, bool] = {"inconclusive": False}
    response_listener: Any = None
    primary_error: BaseException | None = None

    try:
        # Attach network interception before navigation.
        response_listener = _attach_api_interceptor(
            page,
            videos_by_id,
            config.username,
            pending_response_tasks,
            response_state,
            config.request_timeout,
        )

        profile_url = f"https://www.tiktok.com/@{config.username}"
        logger.info("Browser fallback: navigating to %s", profile_url)

        response = await page.goto(
            profile_url,
            wait_until="domcontentloaded",
            timeout=int(config.request_timeout * 1000),
        )
        status = _response_status(response)
        if status == 404:
            raise CreatorNotFoundError(config.username)
        if _status_is_inconclusive(status):
            raise BrowserError(
                f"TikTok profile request returned HTTP {status}; "
                "discovery is inconclusive."
            )
        if _response_url_is_gated(response):
            raise BrowserError(
                "TikTok redirected the profile request to a gated page; "
                "discovery is inconclusive."
            )

        await _wait_for_initial_content(
            page,
            timeout_ms=min(5000, max(1, int(config.request_timeout * 1000))),
        )

        # Do not mistake a challenge or authentication wall for an empty feed.
        await _check_creator_exists(page, config.username)
        await _check_page_access(page)
        profile_identity_verified = await _check_creator_identity(page, config.username)

        # Cookie banners can obscure the feed, but a CAPTCHA/login wall must
        # already have failed above rather than being treated as empty content.
        await _dismiss_overlays(page)
        await _check_page_access(page)

        # ---- Initial hydration data extraction ----
        await _extract_hydration_data(
            page,
            videos_by_id,
            config.username,
            response_state,
        )
        await _await_response_tasks(pending_response_tasks)
        if response_state["inconclusive"] and not videos_by_id:
            # Give already-rendered links one chance to provide an affirmative
            # result, but do not scroll a page whose data endpoint was blocked.
            await _collect_from_dom(page, videos_by_id, config.username)
            if not videos_by_id:
                raise BrowserError(
                    "TikTok returned a blocked or unusable discovery response; "
                    "discovery is inconclusive."
                )

        # ---- Scroll loop ----
        stale_count = 0
        previous_count = len(videos_by_id)

        while True:
            if config.limit is not None and len(videos_by_id) >= config.limit:
                logger.info("Reached video limit (%d). Stopping scroll.", config.limit)
                break

            await _collect_from_dom(page, videos_by_id, config.username)
            if config.limit is not None and len(videos_by_id) >= config.limit:
                logger.info("Reached video limit (%d). Stopping scroll.", config.limit)
                break

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await random_delay(config.scroll_delay_min, config.scroll_delay_max)
            await _await_response_tasks(pending_response_tasks)
            if response_state["inconclusive"] and not videos_by_id:
                raise BrowserError(
                    "TikTok returned a blocked or unusable discovery response; "
                    "discovery is inconclusive."
                )

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

        # Await all response bodies before the last DOM pass. Playwright does
        # not await async event listeners registered with page.on().
        await _await_response_tasks(pending_response_tasks)
        await _collect_from_dom(page, videos_by_id, config.username)
        await _await_response_tasks(pending_response_tasks)
        await _check_page_access(page)

        if response_state["inconclusive"] and not videos_by_id:
            raise BrowserError(
                "TikTok returned a blocked or unusable discovery response; "
                "discovery is inconclusive."
            )
        if not videos_by_id and not profile_identity_verified:
            raise BrowserError(
                "TikTok profile identity could not be confirmed; "
                "discovery is inconclusive."
            )

    except CreatorNotFoundError as exc:
        primary_error = exc
        raise
    except BrowserError as exc:
        primary_error = exc
        raise
    except asyncio.CancelledError as exc:
        primary_error = exc
        raise
    except Exception as exc:
        primary_error = exc
        raise BrowserError(
            f"Browser error while scraping: {_redact_log_value(exc)}"
        ) from exc
    finally:
        await _cleanup_page(
            page,
            response_listener,
            pending_response_tasks,
            primary_error,
        )

    all_videos = list(videos_by_id.values())
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


async def _wait_for_initial_content(page: Page, *, timeout_ms: int = 5000) -> None:
    """Wait briefly for profile content or hydration data to become available."""
    try:
        await page.wait_for_function(
            """() => Boolean(
                document.querySelector('a[href*="/video/"], a[href*="/photo/"]') ||
                window.SIGI_STATE ||
                window.__UNIVERSAL_DATA_FOR_REHYDRATION__ ||
                document.querySelector('script[type="application/json"]')
            )""",
            timeout=timeout_ms,
        )
    except TimeoutError:
        logger.debug("Timed out waiting for initial profile content.")


async def _dismiss_overlays(page: Page) -> None:
    """Dismiss cookie consent banners without bypassing access challenges.

    TikTok often shows overlays that block page interaction. This function
    attempts to click through common consent buttons. Challenge, login, and
    private-account overlays are handled as inconclusive by the access checks.
    """
    dismiss_selectors = [
        # Cookie consent banner buttons
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("Accept cookies")',
        'button:has-text("Allow all")',
        'button:has-text("Decline optional")',
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
    """Raise :class:`CreatorNotFoundError` for explicit missing-account signs.

    Numeric ``404`` text is deliberately not enough: it can be part of a
    post title or another piece of profile content.
    """
    # Check data-e2e attribute — TikTok's most reliable indicator.
    not_found_el = await page.query_selector('[data-e2e="user-not-found"]')
    if not_found_el is not None:
        raise CreatorNotFoundError(username)

    # Only account-specific title phrases identify a missing creator. A title
    # containing the number 404 is not sufficient evidence.
    title = await page.title()
    title_lower = title.casefold() if isinstance(title, str) else ""
    if any(
        phrase in title_lower
        for phrase in (
            "couldn't find this account",
            "could not find this account",
            "this account does not exist",
        )
    ) or re.fullmatch(
        r"(?:tiktok\s*[-:]\s*)?page not found(?:\s*\|\s*tiktok)?",
        title_lower,
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


def _response_status(response: Any) -> int | None:
    """Read a Playwright response status without trusting mock-like values."""
    status = getattr(response, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _status_is_inconclusive(status: int | None) -> bool:
    """Return whether an HTTP response should not be treated as an empty feed."""
    return (
        status in _INCONCLUSIVE_STATUS_CODES
        or (status is not None and 400 <= status <= 499)
        or (status is not None and 500 <= status <= 599)
    )


def _response_url_is_gated(response: Any) -> bool:
    """Detect redirects to login, challenge, or non-TikTok pages."""
    value = getattr(response, "url", None)
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname.casefold() if parsed.hostname else None
    except ValueError:
        return True
    if host is not None and host not in _OFFICIAL_TIKTOK_HOSTS:
        return True
    path = parsed.path.casefold()
    return path.startswith(("/login", "/signup", "/challenge", "/verify", "/captcha"))


def _html_indicates_inconclusive(html: str) -> bool:
    """Recognize strong challenge, login, private, and blocking page markers."""
    if not isinstance(html, str) or not html:
        return False
    lower = html.casefold()
    marker_attributes = " ".join(
        re.findall(
            r"(?:id|class|data-e2e|aria-label|role|action)\s*=\s*"
            r"[\"']([^\"']+)[\"']",
            lower,
        )
    )
    if _BLOCKED_ATTRIBUTE_PATTERN.search(marker_attributes):
        return True
    aria_labels = " ".join(
        re.findall(r"aria-label\s*=\s*[\"']([^\"']+)[\"']", lower)
    )
    if any(
        phrase in aria_labels
        for phrase in (
            "captcha",
            "challenge required",
            "security check",
            "verify you are human",
            "verification required",
        )
    ):
        return True

    visible_markup = re.sub(
        r"<!--.*?-->|<script\b[^>]*>.*?</script\s*>|"
        r"<style\b[^>]*>.*?</style\s*>|<noscript\b[^>]*>.*?</noscript\s*>",
        "",
        lower,
        flags=re.DOTALL,
    )
    visible_text = re.sub(r"<[^>]+>", " ", visible_markup)
    visible_text = " ".join(visible_text.split())
    access_elements = " ".join(
        " ".join(match)
        for match in re.findall(
            r"<(title|h[1-6]|button|form)\b[^>]*>(.*?)</\1\s*>",
            visible_markup,
            flags=re.DOTALL,
        )
    )
    if any(phrase in access_elements for phrase in _BLOCKED_PAGE_PHRASES):
        return True

    # A short, post-less response is usually a bare access wall. Do not scan
    # long visible pages, where a caption or bio may legitimately mention one
    # of these phrases (for example, "login to continue" as ordinary text).
    has_post_link = (
        re.search(
            r"<a\b[^>]*?/(?:video|photo)/[0-9]{15,25}",
            visible_markup,
        )
        is not None
    )
    return (
        not has_post_link
        and len(visible_text) <= 500
        and any(phrase in visible_text for phrase in _BLOCKED_PAGE_PHRASES)
    )


async def _check_page_access(page: Page) -> None:
    """Reject challenge, authentication, private, and blocked profile pages."""
    title = await page.title()
    title_lower = title.casefold() if isinstance(title, str) else ""
    if _title_indicates_inconclusive(title_lower):
        raise BrowserError(
            "TikTok profile page is blocked or requires authentication; "
            "discovery is inconclusive."
        )

    try:
        markup = await page.content()
    except Exception as exc:
        logger.debug(
            "Could not inspect profile page content: %s", _redact_log_value(exc)
        )
        raise BrowserError(
            "Could not inspect TikTok profile access; discovery is inconclusive."
        ) from exc
    if isinstance(markup, str) and _html_indicates_inconclusive(markup):
        raise BrowserError(
            "TikTok profile page is blocked or requires authentication; "
            "discovery is inconclusive."
        )

    # These selectors are intentionally narrow. Generic dialogs and ordinary
    # login links also occur on valid profiles.
    blocked_selectors = (
        '[data-e2e="captcha-verify"]',
        '[data-e2e="login-modal"]',
        '[id*="captcha"]',
        'iframe[src*="captcha"]',
        'form[action*="/login"]',
        '[role="dialog"]:has-text("Verify")',
        '[role="dialog"]:has-text("Log in")',
        '[role="dialog"]:has-text("Private")',
        'button:has-text("Verify")',
        'button:has-text("I\'m not a robot")',
    )
    for selector in blocked_selectors:
        try:
            element = await page.query_selector(selector)
        except Exception:
            continue
        if element is not None:
            try:
                visible = await element.is_visible()
            except Exception:
                visible = True
            if visible:
                raise BrowserError(
                    "TikTok profile page is blocked or requires authentication; "
                    "discovery is inconclusive."
                )


def _title_indicates_inconclusive(title: str) -> bool:
    """Recognize access-wall titles without rejecting usernames like ``login``."""
    title = title.casefold()
    return any(marker in title for marker in _BLOCKED_TITLE_PHRASES)


async def _check_creator_identity(page: Page, username: str) -> bool:
    """Reject a profile page whose explicit title identifies another creator."""
    verified = False
    title = await page.title()
    if not isinstance(title, str):
        return False
    title_match = re.match(
        r"^\s*@?(?P<author>[A-Za-z0-9._]{1,24})"
        r"(?:'s)?(?:\s+profile)?\s*\|\s*tiktok\b",
        title,
        re.IGNORECASE,
    )
    if title_match is None:
        title_match = re.search(
            r"\(@?(?P<author>[A-Za-z0-9._]{1,24})\)\s*\|\s*tiktok\b",
            title,
            re.IGNORECASE,
        )
    if title_match is None:
        title_match = re.search(
            r"(?<![A-Za-z0-9._])@(?P<author>[A-Za-z0-9._]{1,24})\b",
            title,
            re.IGNORECASE,
        )
    if title_match is not None:
        author = title_match.group("author")
        if author.isdigit():
            return False
        verified = True
        if author.casefold() != username.casefold():
            raise BrowserError(
                "TikTok returned a different creator profile; discovery is "
                "inconclusive."
            )

    for selector in ('[data-e2e="user-title"]', 'h1[data-e2e="user-title"]'):
        try:
            element = await page.query_selector(selector)
            if element is None:
                continue
            text = await element.text_content()
        except Exception:
            continue
        author = _valid_author(text.strip()) if isinstance(text, str) else None
        if author is not None and author.casefold() != username.casefold():
            raise BrowserError(
                "TikTok returned a different creator profile; discovery is "
                "inconclusive."
            )
        if author is not None:
            verified = True
    return verified


async def _await_response_tasks(tasks: set[asyncio.Task[Any]]) -> None:
    """Wait for all response handlers, including tasks added while waiting."""
    while tasks:
        current = tuple(tasks)
        results = await asyncio.gather(*current, return_exceptions=True)
        for task in current:
            tasks.discard(task)
        for result in results:
            if isinstance(result, BaseException):
                logger.debug(
                    "TikTok response handler failed: %s",
                    _redact_log_value(result),
                )


async def _maybe_await(value: Any) -> None:
    """Await a value when a test double exposes a synchronous API as async."""
    if inspect.isawaitable(value):
        await value


async def _cleanup_page(
    page: Page,
    response_listener: Any,
    pending_tasks: set[asyncio.Task[Any]],
    primary_error: BaseException | None,
) -> None:
    """Detach and close a page without hiding an exception from scraping."""
    cleanup_errors: list[BaseException] = []

    if response_listener is not None:
        try:
            remove_result = page.remove_listener(  # type: ignore[func-returns-value]
                "response",
                response_listener,
            )
            await _maybe_await(remove_result)
        except BaseException as exc:
            cleanup_errors.append(exc)

    try:
        await _await_response_tasks(pending_tasks)
    except BaseException as exc:
        cleanup_errors.append(exc)

    try:
        await page.close()
    except BaseException as exc:
        cleanup_errors.append(exc)

    if not cleanup_errors:
        return
    if primary_error is not None:
        logger.warning(
            "Browser page cleanup failed after the primary error: %s",
            _redact_log_value(cleanup_errors[0]),
        )
        return
    raise BrowserError("Browser page cleanup failed.") from cleanup_errors[0]


async def _cleanup_browser_resources(
    playwright: Any,
    browser: Any,
    context: Any,
    primary_error: BaseException | None,
) -> None:
    """Close owned browser resources in order while preserving primary errors."""
    cleanup_errors: list[BaseException] = []
    for resource, method_name in (
        (context, "close"),
        (browser, "close"),
        (playwright, "stop"),
    ):
        try:
            await _maybe_await(getattr(resource, method_name)())
        except BaseException as exc:
            cleanup_errors.append(exc)

    if not cleanup_errors:
        return
    if primary_error is not None:
        logger.warning(
            "Browser resource cleanup failed after the primary error: %s",
            _redact_log_value(cleanup_errors[0]),
        )
        return
    raise BrowserError("Browser resource cleanup failed.") from cleanup_errors[0]


def _attach_api_interceptor(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
    requested_username: str | None = None,
    pending_tasks: set[asyncio.Task[Any]] | None = None,
    response_state: dict[str, bool] | None = None,
    response_timeout: float | None = None,
) -> Any:
    """Listen for item-list responses and retain tasks for later awaiting."""
    tracked_tasks = pending_tasks if pending_tasks is not None else set()
    state = response_state if response_state is not None else {"inconclusive": False}
    response_lock = asyncio.Lock()

    async def _process_response_unlocked(response: Response) -> None:
        response_url = getattr(response, "url", "")
        if not isinstance(response_url, str):
            return
        response_url_lower = response_url.casefold()
        if not any(
            fragment in response_url_lower for fragment in _API_ITEM_LIST_FRAGMENTS
        ):
            return
        try:
            response_host = urlsplit(response_url).hostname
        except ValueError:
            return
        if (
            response_host is None
            or response_host.casefold() not in _OFFICIAL_TIKTOK_HOSTS
        ):
            return

        status = _response_status(response)
        if _status_is_inconclusive(status) or (
            status is not None and not 200 <= status <= 299
        ):
            state["inconclusive"] = True
            return

        try:
            body = (
                await asyncio.wait_for(response.body(), timeout=response_timeout)
                if response_timeout is not None
                else await response.body()
            )
            data = json.loads(body)
            if not isinstance(data, dict):
                state["inconclusive"] = True
                return
            if _hydration_indicates_inconclusive(data):
                state["inconclusive"] = True
                return

            payload_status = data.get("status_code", data.get("statusCode"))
            if isinstance(payload_status, str) and payload_status.isdigit():
                payload_status = int(payload_status)
            if isinstance(payload_status, int) and _status_is_inconclusive(
                payload_status
            ):
                state["inconclusive"] = True
                return

            for key in ("status_msg", "statusMsg", "message", "msg", "error"):
                payload_message = data.get(key)
                if not isinstance(payload_message, str):
                    continue
                if any(
                    phrase in payload_message.casefold()
                    for phrase in (
                        "blocked",
                        "captcha",
                        "challenge",
                        "forbidden",
                        "login",
                        "private",
                        "too many requests",
                    )
                ):
                    state["inconclusive"] = True
                    return

            if "itemList" in data:
                item_list: Any = data["itemList"]
            elif isinstance(data.get("data"), dict) and "itemList" in data["data"]:
                item_list = data["data"]["itemList"]
            else:
                state["inconclusive"] = True
                return
            if not isinstance(item_list, list):
                state["inconclusive"] = True
                return
            for item in item_list:
                if isinstance(item, dict):
                    _add_record(
                        videos_by_id,
                        item,
                        requested_username,
                        allow_missing_url=True,
                    )
        except Exception as exc:
            # DOM/hydration remain useful, but an unusable target response
            # must not be reported as a successful empty profile.
            state["inconclusive"] = True
            logger.debug(
                "Failed to parse TikTok item-list response: %s",
                _redact_log_value(exc),
            )

    async def _process_response(response: Response) -> None:
        # Preserve response arrival order even when an earlier response body
        # takes longer to become available than a later one.
        async with response_lock:
            await _process_response_unlocked(response)

    def _on_response(response: Response) -> asyncio.Task[Any]:
        # Playwright's event emitter does not await async listeners. Returning
        # the task keeps direct unit-test invocation awaitable as well.
        task = asyncio.create_task(_process_response(response))
        tracked_tasks.add(task)
        task.add_done_callback(tracked_tasks.discard)
        return task

    page.on("response", _on_response)
    return _on_response


async def _extract_hydration_data(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
    requested_username: str | None = None,
    response_state: dict[str, bool] | None = None,
) -> None:
    """Extract video data from globals and server-side hydration scripts."""
    try:

        def _consume(candidate: Any) -> None:
            if response_state is not None and _hydration_indicates_inconclusive(
                candidate
            ):
                response_state["inconclusive"] = True
            if isinstance(candidate, dict):
                _parse_hydration_items(candidate, videos_by_id, requested_username)
                return
            if not isinstance(candidate, str) or not candidate.strip():
                return
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return
            if isinstance(parsed, (dict, list)) or (
                isinstance(parsed, str) and parsed != candidate
            ):
                _consume(parsed)

        raw_states: Any = await page.evaluate("""
            () => {
                const states = [];
                for (const key of [
                    'SIGI_STATE',
                    '__UNIVERSAL_DATA_FOR_REHYDRATION__',
                ]) {
                    try {
                        if (window[key]) states.push(JSON.stringify(window[key]));
                    } catch (_) {
                        // A malformed global must not prevent script parsing.
                    }
                }

                for (const script of document.scripts) {
                    const id = (script.id || '').toLowerCase();
                    const type = (script.type || '').toLowerCase();
                    if (
                        type === 'application/json' ||
                        id.includes('sigi') ||
                        id.includes('universal') ||
                        id.includes('hydration') ||
                        id.includes('rehydration') ||
                        id === '__next_data__'
                    ) {
                        const text = script.textContent;
                        if (text) states.push(text);
                    }
                }
                return states;
            }
        """)
        if raw_states is None:
            candidates: list[Any] = []
        else:
            candidates = raw_states if isinstance(raw_states, list) else [raw_states]
        for candidate in candidates:
            _consume(candidate)

        # Keep a Python-side path as well. It covers pages that expose state in
        # script elements but not on window globals and makes malformed scripts
        # independently ignorable.
        script_elements = await page.query_selector_all("script")
        if isinstance(script_elements, Iterable):
            for script in script_elements:
                try:
                    script_text = await script.text_content()
                except Exception:
                    continue
                _consume(script_text)

    except Exception as exc:
        logger.debug(
            "Hydration data extraction failed: %s",
            _redact_log_value(exc),
        )


def _parse_hydration_items(
    data: dict[str, Any],
    videos_by_id: dict[str, VideoInfo],
    requested_username: str | None = None,
) -> None:
    """Walk a hydration-state dict and pull out validated video items."""

    def _parse_item_module(item_module: Any) -> None:
        if not isinstance(item_module, dict):
            return
        for post_id, item_data in item_module.items():
            if not isinstance(item_data, dict):
                continue
            record = dict(item_data)
            record.setdefault("id", post_id)
            _add_record(
                videos_by_id,
                record,
                requested_username,
                allow_missing_url=True,
            )

    def _parse_item_list(item_list: Any) -> None:
        if isinstance(item_list, list):
            for item in item_list:
                if isinstance(item, dict):
                    _add_record(
                        videos_by_id,
                        item,
                        requested_username,
                        allow_missing_url=True,
                    )
        elif isinstance(item_list, dict):
            _parse_item_module(item_list)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "ItemModule":
                    _parse_item_module(child)
                elif key == "itemList":
                    _parse_item_list(child)
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(data)


async def _collect_from_dom(
    page: Page,
    videos_by_id: dict[str, VideoInfo],
    requested_username: str | None = None,
) -> None:
    """Query the DOM for video and photo links and add them to *videos_by_id*."""
    selector = f"{_VIDEO_LINK_SELECTOR}, {_PHOTO_LINK_SELECTOR}"
    links = await page.query_selector_all(selector)
    for link in links:
        href = await link.get_attribute("href")
        post = _canonical_post_url(href, requested_username)
        if post is None:
            continue
        post_id, kind, author = post
        if post_id in videos_by_id:
            continue
        videos_by_id[post_id] = VideoInfo(
            video_id=post_id,
            url=f"https://www.tiktok.com/@{author}/{kind}/{post_id}",
        )
