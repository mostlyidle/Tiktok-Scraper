"""Video and photo download via yt-dlp with concurrency control."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import functools
import http.client
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import yt_dlp
from curl_cffi import requests as curl_requests
from yt_dlp.networking.impersonate import ImpersonateTarget

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.models import ScrapeResult, VideoInfo
from tiktok_scraper.utils import (
    RequestLimiter,
    bounded_backoff,
    check_ffmpeg,
    is_completed_artifact,
    is_completed_slideshow_artifact,
    is_valid_video_id,
    isolated_cookie_file,
    parse_retry_after,
    post_lock,
    read_netscape_cookie_pairs,
    redact_log_value,
    safe_output_path,
    validate_output_dir,
)

logger = logging.getLogger(__name__)


class _YtdlpLogger:
    """Route yt-dlp output through this module's logger."""

    def debug(self, message: str) -> None:
        logger.debug("yt-dlp: %s", redact_log_value(message))

    def warning(self, message: str) -> None:
        logger.warning("yt-dlp: %s", redact_log_value(message))

    def error(self, message: str) -> None:
        logger.error("yt-dlp: %s", redact_log_value(message))


_MAX_CONCURRENT_IMAGE_DOWNLOADS = 3
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_IMPERSONATE_CHROME = ImpersonateTarget.from_str("chrome")
_YTDLP_LOGGER = _YtdlpLogger()
_TIKTOK_HOSTS = frozenset(
    {
        "tiktok.com",
        "www.tiktok.com",
        "m.tiktok.com",
        "tiktokv.com",
        "www.tiktokv.com",
    }
)
_POST_PATH_PATTERN = re.compile(
    r"^/@(?P<author>[A-Za-z0-9._]{1,24})/"
    r"(?P<kind>video|photo)/(?P<id>[0-9]{15,25})/?$",
    re.IGNORECASE,
)
_AUTHOR_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,24}$")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FRONTITY_STATE_PATTERN = re.compile(
    r'<script[^>]+id=["\']__FRONTITY_CONNECT_STATE__["\'][^>]*>'
    r"(?P<data>.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)


class _DownloadOutcome(enum.Enum):
    """Result of a single download attempt."""

    VIDEO = "video"
    AUDIO = "audio"
    PHOTOS = "photos"
    SKIPPED = "skipped"
    SKIPPED_AUDIO_ONLY = "skipped_audio_only"
    FAILED = "failed"


class _FailureKind(enum.Enum):
    """Classification used to decide whether a failed workflow is retried."""

    PERMANENT = "permanent"
    TRANSIENT = "transient"
    LOCAL = "local"


class MetadataMismatchError(ValueError):
    """Raised when a post URL resolves to a different or missing ID."""


@dataclass(frozen=True, slots=True)
class _DownloadFailure:
    """An internal failure that retains retry classification."""

    error: Exception
    kind: _FailureKind


_DownloadResult = _DownloadOutcome | _DownloadFailure


async def download_videos(
    videos: list[VideoInfo],
    config: ScraperConfig,
) -> ScrapeResult:
    """Download a list of TikTok videos (and photo slideshows) using yt-dlp.

    Already-downloaded content (detected by file-name or directory matching
    on video ID) is automatically skipped.  Downloads are limited to
    ``config.max_concurrent_downloads`` at a time via a fixed-size worker
    pool.

    When a post is a photo slideshow (no video stream), the individual
    images are downloaded directly.  If ``config.video_only`` is ``True``,
    audio-only posts that have no images are skipped entirely.

    Args:
        videos: The video list produced by the scraper.
        config: Scraper configuration.

    Returns:
        A :class:`ScrapeResult` summarising the session.
    """
    validate_output_dir(config.output_dir, create=True)

    unique_videos = _deduplicate_videos(videos, config.username)
    result = ScrapeResult(
        username=config.username,
        total_found=len(unique_videos),
    )

    to_download: list[VideoInfo] = []
    for video in unique_videos:
        if _has_completed_download(video, config.output_dir):
            result.skipped += 1
        else:
            to_download.append(video)

    if result.skipped:
        logger.info("Skipping %d already-downloaded videos.", result.skipped)

    if not to_download:
        logger.info("All videos already downloaded. Nothing to do.")
        return result

    logger.info(
        "Downloading %d videos (concurrency=%d)...",
        len(to_download),
        config.max_concurrent_downloads,
    )

    image_semaphore = asyncio.Semaphore(
        min(config.max_concurrent_downloads, _MAX_CONCURRENT_IMAGE_DOWNLOADS)
    )
    request_limiter = RequestLimiter(config.request_delay)
    queue: asyncio.Queue[tuple[int, VideoInfo] | None] = asyncio.Queue()
    outcomes: list[_DownloadResult | None] = [None] * len(to_download)

    for index, video in enumerate(to_download):
        queue.put_nowait((index, video))

    async def _worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return

                outcome_index, video = item
                try:
                    async with post_lock(config.output_dir, video.video_id):
                        if _has_completed_download(video, config.output_dir):
                            outcomes[outcome_index] = _DownloadOutcome.SKIPPED
                        else:
                            outcomes[outcome_index] = await _download_single(
                                video,
                                config,
                                outcome_index + 1,
                                len(to_download),
                                image_semaphore,
                                request_limiter,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    outcomes[outcome_index] = _DownloadFailure(
                        exc,
                        _classify_download_error(exc),
                    )
            finally:
                queue.task_done()

    worker_count = min(config.max_concurrent_downloads, len(to_download))
    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    try:
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    for video, outcome in zip(to_download, outcomes, strict=True):
        if isinstance(outcome, _DownloadFailure):
            result.failed += 1
            result.failed_videos.append(video.video_id)
            result.failed_reasons[video.video_id] = redact_log_value(str(outcome.error))
            logger.error(
                "Download failed for %s (%s): %s",
                video.video_id,
                outcome.kind.value,
                redact_log_value(outcome.error),
            )
        elif outcome is _DownloadOutcome.VIDEO:
            result.downloaded += 1
        elif outcome is _DownloadOutcome.AUDIO:
            result.audio_downloaded += 1
        elif outcome is _DownloadOutcome.PHOTOS:
            result.photos_downloaded += 1
        elif outcome is _DownloadOutcome.SKIPPED:
            result.skipped += 1
        elif outcome is _DownloadOutcome.SKIPPED_AUDIO_ONLY:
            result.skipped_audio_only += 1
        else:
            result.failed += 1
            result.failed_videos.append(video.video_id)
            result.failed_reasons[video.video_id] = "unknown download failure"

    return result


def _deduplicate_videos(videos: list[VideoInfo], username: str) -> list[VideoInfo]:
    """Keep first-seen posts and reject records from another creator."""
    by_id: dict[str, VideoInfo] = {}
    for video in videos:
        if not isinstance(video, VideoInfo):
            raise TypeError("videos must contain VideoInfo instances")
        by_id.setdefault(video.video_id, video)
    wrong_creator = [
        video.video_id
        for video in by_id.values()
        if video.username.casefold() != username.casefold()
    ]
    if wrong_creator:
        raise ValueError(
            f"discovered posts do not belong to @{username}: {', '.join(wrong_creator)}"
        )
    return list(by_id.values())


def _has_completed_download(video: VideoInfo, output_dir: Path) -> bool:
    """Check an existing artifact using the post's known content type."""
    slideshow_dir = safe_output_path(output_dir, video.video_id)
    if is_completed_slideshow_artifact(slideshow_dir):
        return True
    if video.is_photo:
        return False
    for media_type in ("video", "audio"):
        artifact = _find_completed_artifact(
            output_dir,
            video.video_id,
            media_type=media_type,
        )
        if artifact is not None and _resume_artifact_matches_type(
            artifact,
            output_dir,
            video.video_id,
            media_type,
        ):
            return True
    return False


def _resume_artifact_matches_type(
    artifact: Path,
    output_dir: Path,
    video_id: str,
    media_type: str,
) -> bool:
    """Disambiguate WebM resume files using the type recorded at download time."""
    if artifact.suffix.casefold() != ".webm":
        return True
    marker = safe_output_path(output_dir, f".{video_id}.webm-type")
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="ascii").strip() == media_type
    except (OSError, UnicodeError):
        return False


def _metadata_author(value: Any) -> str | None:
    """Read a creator handle from yt-dlp's author-shaped metadata."""
    if isinstance(value, str):
        author = value.strip().removeprefix("@")
        return author if _AUTHOR_PATTERN.fullmatch(author) else None
    if not isinstance(value, dict):
        return None
    for key in (
        "uniqueId",
        "unique_id",
        "username",
        "userName",
        "authorName",
        "author_name",
    ):
        nested_author = _metadata_author(value.get(key))
        if nested_author is not None:
            return nested_author
    for key in ("author", "authorInfo", "authorInfos", "user"):
        nested_author = _metadata_author(value.get(key))
        if nested_author is not None:
            return nested_author
    return None


def _metadata_authors(value: Any) -> tuple[set[str], bool]:
    """Collect all username aliases from resolved metadata."""
    if isinstance(value, str):
        author = _metadata_author(value)
        return ({author.casefold()} if author is not None else set(), author is None)
    if not isinstance(value, dict):
        return set(), True

    authors: set[str] = set()
    invalid = False
    seen_alias = False
    for key in (
        "uniqueId",
        "unique_id",
        "username",
        "userName",
        "authorName",
        "author_name",
    ):
        if key not in value:
            continue
        seen_alias = True
        author = _metadata_author(value[key])
        if author is None:
            invalid = True
        else:
            authors.add(author.casefold())
    for key in ("author", "authorInfo", "authorInfos", "user"):
        if key not in value:
            continue
        seen_alias = True
        nested_authors, nested_invalid = _metadata_authors(value[key])
        authors.update(nested_authors)
        invalid = invalid or nested_invalid
    return authors, invalid or (seen_alias and not authors)


def _metadata_post_parts(value: Any) -> tuple[str, str] | None:
    """Return ``(post_id, author)`` for a strict official TikTok URL."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname.casefold() if parsed.hostname else None
        if (
            parsed.scheme.casefold() != "https"
            or hostname not in _TIKTOK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            return None
    except (TypeError, ValueError):
        return None
    match = _POST_PATH_PATTERN.fullmatch(parsed.path)
    if match is None:
        return None
    return match.group("id"), match.group("author")


def _is_matching_embed_url(value: Any, video_id: str) -> bool:
    """Return whether a metadata URL is the exact requested embed route."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        _is_official_page_url(value)
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and parsed.path.rstrip("/") == f"/embed/v2/{video_id}"
    )


def _is_official_page_url(value: Any) -> bool:
    """Return whether a URL is an HTTPS page on a known TikTok host."""
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
        and hostname in _TIKTOK_HOSTS
        and parsed.username is None
        and parsed.password is None
        and port is None
    )


def _metadata_matches_video(
    info: dict[str, Any],
    video: VideoInfo,
    *,
    require_identity: bool = True,
) -> bool:
    """Ensure resolved metadata still identifies the requested post/creator."""
    expected_id = video.video_id
    expected_author = video.username.casefold()
    identity_seen = False

    for key in ("webpage_url", "webpageUrl", "original_url", "originalUrl"):
        value = info.get(key)
        if value is None or value == "":
            continue
        parts = _metadata_post_parts(value)
        if parts is None:
            if _is_matching_embed_url(value, expected_id):
                continue
            return False
        identity_seen = True
        if parts[0] != expected_id:
            return False
        if parts[1].casefold() != expected_author:
            return False

    uploader = info.get("uploader")
    if uploader not in (None, ""):
        identity_seen = True
        uploader_authors, uploader_invalid = _metadata_authors(uploader)
        if uploader_invalid or uploader_authors != {expected_author}:
            return False
    else:
        for key in ("author", "authorInfo", "authorInfos"):
            if key not in info or info[key] in (None, ""):
                continue
            identity_seen = True
            authors, invalid = _metadata_authors(info[key])
            if invalid or authors != {expected_author}:
                return False

    return identity_seen or not require_identity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _download_single(
    video: VideoInfo,
    config: ScraperConfig,
    index: int,
    total: int,
    image_semaphore: asyncio.Semaphore,
    request_limiter: RequestLimiter | None = None,
) -> _DownloadResult:
    """Download a single post with retry/backoff around the full workflow.

    Retries re-run metadata extraction and download from scratch, which helps
    recover from transient TikTok and network failures.

    Returns:
        The final outcome after all attempts.
    """
    max_attempts = max(1, config.download_retries)

    for attempt in range(1, max_attempts + 1):
        outcome = await _download_single_once(
            video,
            config,
            index,
            total,
            image_semaphore,
            request_limiter,
        )
        if not isinstance(outcome, _DownloadFailure):
            return outcome
        if outcome.kind is not _FailureKind.TRANSIENT or attempt >= max_attempts:
            return outcome

        delay = bounded_backoff(
            attempt,
            base=config.retry_backoff_base,
            cap=config.retry_backoff_max,
            retry_after=_retry_after_from_exception(outcome.error),
        )
        logger.warning(
            "[%d/%d] Retrying %s (attempt %d/%d) in %.1fs ...",
            index,
            total,
            video.video_id,
            attempt + 1,
            max_attempts,
            delay,
        )
        await asyncio.sleep(delay)

    logger.error(
        "[%d/%d] Exhausted retries for %s after %d attempts.",
        index,
        total,
        video.video_id,
        max_attempts,
    )
    return _DownloadFailure(
        RuntimeError("download retry loop exhausted"),
        _FailureKind.TRANSIENT,
    )


async def _download_single_once(
    video: VideoInfo,
    config: ScraperConfig,
    index: int,
    total: int,
    image_semaphore: asyncio.Semaphore,
    request_limiter: RequestLimiter | None = None,
) -> _DownloadResult:
    """Download a single post.

    Inspects post metadata first to determine content type, then:
    - Video posts: download via yt-dlp (existing behaviour).
    - Photo slideshows: download images directly via urllib.
    - Audio-only posts (no images): skip when ``video_only``, otherwise
      download via yt-dlp.

    Returns:
        The outcome of the download.
    """
    logger.info("[%d/%d] Inspecting %s ...", index, total, video.video_id)

    loop = asyncio.get_running_loop()
    extract_opts = _build_extract_opts(config)

    try:
        info = await _retry_operation(
            lambda: loop.run_in_executor(
                None,
                functools.partial(_sync_extract_info, video.url, extract_opts),
            ),
            config,
            "metadata extraction",
            request_limiter=request_limiter,
        )
    except Exception as exc:
        logger.warning(
            "[%d/%d] Primary extraction failed for %s: %s",
            index,
            total,
            video.video_id,
            redact_log_value(exc),
        )
        failure = _failed_download(exc)
        if failure.kind is _FailureKind.LOCAL:
            return failure
        if video.is_photo:
            return failure
        return await _download_embed_fallback(
            video, config, index, total, request_limiter
        )

    if info is None:
        logger.warning(
            "[%d/%d] No metadata returned for %s.",
            index,
            total,
            video.video_id,
        )
        if video.is_photo:
            return _failed_download(
                DownloadArtifactError("photo post returned no metadata")
            )
        return await _download_embed_fallback(video, config, index, total, request_limiter)
    info_id = str(info.get("id", ""))
    if info_id != video.video_id or not is_valid_video_id(info_id):
        return _DownloadFailure(
            MetadataMismatchError(
                "yt-dlp metadata ID does not match the requested post"
            ),
            _FailureKind.LOCAL,
        )
    if not _metadata_matches_video(info, video):
        return _DownloadFailure(
            MetadataMismatchError(
                "yt-dlp metadata does not identify the requested creator"
            ),
            _FailureKind.LOCAL,
    )

    outcome = await _download_from_info(
        video,
        info,
        config,
        index,
        total,
        image_semaphore,
        request_limiter,
    )
    if (
        isinstance(outcome, _DownloadFailure)
        and outcome.kind is not _FailureKind.LOCAL
        and _has_video_formats(info)
    ):
        return await _download_embed_fallback(
            video, config, index, total, request_limiter
        )
    return outcome


async def _download_embed_fallback(
    video: VideoInfo,
    config: ScraperConfig,
    index: int,
    total: int,
    request_limiter: RequestLimiter | None = None,
) -> _DownloadResult:
    """Download a video through TikTok's embed page when post extraction fails."""
    logger.warning(
        "[%d/%d] Falling back to TikTok embed for %s.",
        index,
        total,
        video.video_id,
    )
    output_path = safe_output_path(config.output_dir, f"{video.video_id}.mp4")
    loop = asyncio.get_running_loop()
    embed_opts = _build_embed_ydl_opts(config, video.video_id)
    embed_opts["_expected_author"] = video.username

    try:
        metadata = await _retry_operation(
            lambda: loop.run_in_executor(
                None,
                functools.partial(
                    _sync_download_embed,
                    video.video_id,
                    embed_opts,
                ),
            ),
            config,
            f"embed {video.video_id}",
            request_limiter=request_limiter,
        )
        if isinstance(metadata, dict):
            metadata_id = str(metadata.get("id", ""))
            if (
                metadata_id != video.video_id
                or not is_valid_video_id(metadata_id)
                or not _metadata_matches_video(
                    metadata,
                    video,
                    require_identity=False,
                )
            ):
                raise MetadataMismatchError(
                    "TikTok embed metadata does not identify the requested post"
                )
        if not is_completed_artifact(output_path, media_type="video"):
            raise DownloadArtifactError(
                f"TikTok embed did not create {output_path.name}"
            )
    except Exception as exc:
        logger.error(
            "[%d/%d] Embed fallback failed for %s: %s",
            index,
            total,
            video.video_id,
            redact_log_value(exc),
        )
        return _failed_download(exc)

    logger.info("[%d/%d] Completed embed fallback for %s.", index, total, video.video_id)
    return _DownloadOutcome.VIDEO


async def _download_from_info(
    video: VideoInfo,
    info: dict[str, Any],
    config: ScraperConfig,
    index: int,
    total: int,
    image_semaphore: asyncio.Semaphore,
    request_limiter: RequestLimiter | None = None,
) -> _DownloadResult:
    """Download a post from a resolved yt-dlp metadata payload."""
    has_video = _has_video_formats(info)
    loop = asyncio.get_running_loop()
    is_photo_post = video.is_photo or _info_is_photo_post(info)
    has_existing_slideshow = is_completed_slideshow_artifact(
        safe_output_path(config.output_dir, video.video_id)
    )
    if has_existing_slideshow and (is_photo_post or not has_video):
        return _DownloadOutcome.SKIPPED
    image_urls = _extract_image_urls(info, is_photo_post=is_photo_post)

    # A photo URL/content marker takes precedence over generic media formats.
    # Never turn an identified slideshow into an audio-only download.
    if is_photo_post and not image_urls:
        return _failed_download(
            DownloadArtifactError("photo post did not expose any images")
        )

    if has_video and not is_photo_post:
        # Standard video post — download via yt-dlp.
        logger.info("[%d/%d] Downloading video %s ...", index, total, video.video_id)
        ffmpeg_available = check_ffmpeg()
        if _requires_ffmpeg(info) and not ffmpeg_available:
            logger.error(
                "[%d/%d] ffmpeg is required to merge video and audio for %s.",
                index,
                total,
                video.video_id,
            )
            return _failed_download(
                DownloadArtifactError("ffmpeg is required for this download")
            )
        ydl_opts = _build_ydl_opts(
            config,
            merge=ffmpeg_available,
            video_id=video.video_id,
        )
        try:
            artifact_valid = await _retry_operation(
                lambda: loop.run_in_executor(
                    None,
                    functools.partial(
                        _sync_download_info,
                        info,
                        ydl_opts,
                        video.video_id,
                        "video",
                    ),
                ),
                config,
                f"video {video.video_id}",
                request_limiter=request_limiter,
            )
            if artifact_valid is not True:
                raise DownloadArtifactError(video.video_id)
            logger.info("[%d/%d] Completed video %s.", index, total, video.video_id)
            return _DownloadOutcome.VIDEO
        except Exception as exc:
            logger.warning(
                "[%d/%d] Primary video download failed for %s: %s",
                index,
                total,
                video.video_id,
                redact_log_value(exc),
            )
            return _failed_download(exc)

    if image_urls:
        logger.info(
            "[%d/%d] Downloading %d slideshow images for %s ...",
            index,
            total,
            len(image_urls),
            video.video_id,
        )
        try:
            await _download_images(
                image_urls,
                config.output_dir,
                video.video_id,
                image_semaphore,
                # The post-level loop owns retries for the full workflow.
                retries=1,
                retry_backoff_base=config.retry_backoff_base,
                retry_backoff_max=config.retry_backoff_max,
                request_limiter=request_limiter,
                cookies_file=config.cookies_file,
                timeout=config.request_timeout,
                max_bytes=config.image_max_bytes,
            )
            logger.info(
                "[%d/%d] Completed slideshow %s (%d images).",
                index,
                total,
                video.video_id,
                len(image_urls),
            )
            return _DownloadOutcome.PHOTOS
        except Exception as exc:
            logger.error(
                "[%d/%d] Failed slideshow %s: %s",
                index,
                total,
                video.video_id,
                redact_log_value(exc),
            )
            return _failed_download(exc)

    # Audio-only post with no images.
    if config.video_only:
        logger.info(
            "[%d/%d] Skipping audio-only post %s (--video-only).",
            index,
            total,
            video.video_id,
        )
        return _DownloadOutcome.SKIPPED_AUDIO_ONLY

    # Download the audio via yt-dlp (default behaviour).
    logger.info(
        "[%d/%d] Downloading audio-only post %s ...",
        index,
        total,
        video.video_id,
    )
    ydl_opts = _build_ydl_opts(config, merge=False, video_id=video.video_id)
    try:
        artifact_valid = await _retry_operation(
            lambda: loop.run_in_executor(
                None,
                functools.partial(
                    _sync_download_info,
                    info,
                    ydl_opts,
                    video.video_id,
                    "audio",
                ),
            ),
            config,
            f"audio {video.video_id}",
            request_limiter=request_limiter,
        )
        if artifact_valid is not True:
            raise DownloadArtifactError(video.video_id)
        logger.info("[%d/%d] Completed audio %s.", index, total, video.video_id)
        return _DownloadOutcome.AUDIO
    except Exception as exc:
        logger.error(
            "[%d/%d] Failed audio %s: %s",
            index,
            total,
            video.video_id,
            redact_log_value(exc),
        )
        return _failed_download(exc)


# ---------------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------------


def _is_photo_post_url(url: str) -> bool:
    """Return whether a URL is an official TikTok photo-post URL."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname.casefold() if parsed.hostname else None
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme.casefold() == "https"
        and hostname in _TIKTOK_HOSTS
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and (
            match := _POST_PATH_PATTERN.fullmatch(parsed.path)
        ) is not None
        and match.group("kind").casefold() == "photo"
    )


def _frontity_video_data(value: Any, video_id: str) -> dict[str, Any] | None:
    """Find matching Frontity video data in a decoded hydration payload."""
    if isinstance(value, dict):
        candidate = value.get("videoData")
        if isinstance(candidate, dict):
            item_info = candidate.get("itemInfos")
            if isinstance(item_info, dict) and str(item_info.get("id", "")) == video_id:
                return candidate
        for child in value.values():
            result = _frontity_video_data(child, video_id)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _frontity_video_data(child, video_id)
            if result is not None:
                return result
    return None


def _sync_extract_photo_info(
    url: str,
    opts: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract photo-post metadata from TikTok's embed hydration state."""
    try:
        parsed = urlsplit(url)
        match = _POST_PATH_PATTERN.fullmatch(parsed.path)
    except (TypeError, ValueError):
        return None
    if match is None or match.group("kind").casefold() != "photo":
        return None
    video_id = match.group("id")
    embed_url = f"https://www.tiktok.com/embed/v2/{video_id}"
    cookiefile = opts.get("cookiefile")
    cookies_file = Path(cookiefile) if isinstance(cookiefile, str) else None
    response = curl_requests.get(
        embed_url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.tiktok.com/",
        },
        cookies=read_netscape_cookie_pairs(cookies_file, url=embed_url),
        impersonate="chrome",
        timeout=float(opts.get("socket_timeout", 30.0)),
        allow_redirects=False,
    )
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 300 <= status < 400:
        raise yt_dlp.utils.DownloadError("TikTok photo embed returned a redirect")
    final_url = getattr(response, "url", embed_url)
    if not _is_official_page_url(final_url):
        raise yt_dlp.utils.DownloadError("TikTok photo embed returned an untrusted URL")
    response.raise_for_status()  # type: ignore[no-untyped-call]
    html = response.text
    state_match = _FRONTITY_STATE_PATTERN.search(html)
    if state_match is None:
        raise yt_dlp.utils.DownloadError("TikTok photo embed had no hydration state")
    try:
        state = json.loads(state_match.group("data"))
    except json.JSONDecodeError as exc:
        raise yt_dlp.utils.DownloadError(
            "TikTok photo embed hydration was invalid"
        ) from exc
    video_data = _frontity_video_data(state, video_id)
    if video_data is None:
        raise yt_dlp.utils.DownloadError("TikTok photo embed had no matching post")
    author = _metadata_author(video_data.get("authorInfos"))
    result: dict[str, Any] = {
        "id": video_id,
        "webpage_url": url,
        "formats": [],
        "vcodec": "none",
        "imagePost": video_data.get("imagePostInfo"),
    }
    if author is not None:
        result["uploader"] = author
    return result


def _photo_variant_url(url: str) -> str | None:
    """Convert a canonical video URL into its matching photo route."""
    if not _is_official_page_url(url):
        return None
    try:
        parsed = urlsplit(url)
        match = _POST_PATH_PATTERN.fullmatch(parsed.path)
    except (TypeError, ValueError):
        return None
    if match is None or match.group("kind").casefold() != "video":
        return None
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/@{match.group('author')}/photo/{match.group('id')}",
            "",
            "",
        )
    )


def _sync_try_extract_photo_info(
    url: str,
    opts: dict[str, Any],
) -> dict[str, Any] | None:
    """Probe the photo route without masking the primary extraction result."""
    photo_url = _photo_variant_url(url)
    if photo_url is None:
        return None
    try:
        info = _sync_extract_photo_info(photo_url, opts)
    except Exception as exc:
        logger.debug("Photo-route probe failed: %s", redact_log_value(exc))
        return None
    return info if info is not None and _info_is_photo_post(info) else None


def _sync_extract_info(
    url: str,
    opts: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract post metadata without downloading (runs in an executor)."""
    if _is_photo_post_url(url):
        return _sync_extract_photo_info(url, opts)
    try:
        with _isolated_ydl_opts(opts) as safe_opts, yt_dlp.YoutubeDL(safe_opts) as ydl:
            result = ydl.extract_info(url, download=False)
    except Exception:
        photo_info = _sync_try_extract_photo_info(url, opts)
        if photo_info is not None:
            return photo_info
        raise
    if result is None:
        return None
    info = dict(result)
    if not _has_video_formats(info) and not _has_audio_formats(info):
        photo_info = _sync_try_extract_photo_info(url, opts)
        if photo_info is not None:
            return photo_info
    return info


class DownloadArtifactError(RuntimeError):
    """Raised when yt-dlp returns without producing a completed artifact."""


@contextlib.contextmanager
def _isolated_ydl_opts(opts: dict[str, Any]) -> Any:
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


@contextlib.contextmanager
def _staged_ydl_opts(
    opts: dict[str, Any],
    artifact_id: str,
) -> Iterator[tuple[dict[str, Any], Path | None, Path | None]]:
    """Redirect yt-dlp writes into a private staging directory."""
    outtmpl = opts.get("outtmpl")
    if not isinstance(outtmpl, str) or not is_valid_video_id(artifact_id):
        yield opts, None, None
        return

    paths = opts.get("paths")
    if isinstance(paths, dict) and isinstance(paths.get("home"), str):
        output_dir = validate_output_dir(Path(paths["home"]))
    elif Path(outtmpl).is_absolute():
        output_dir = validate_output_dir(Path(outtmpl).parent)
    else:
        output_dir = validate_output_dir(Path.cwd())

    template_path = Path(outtmpl)
    if template_path.is_absolute():
        try:
            template_path = template_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError("yt-dlp output template escapes output_dir") from exc
    safe_output_path(output_dir, template_path)

    staging_dir = safe_output_path(output_dir, f".{artifact_id}.download")
    _remove_staging_path(staging_dir)
    staging_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
    staged_opts = dict(opts)
    staged_paths = dict(paths) if isinstance(paths, dict) else {}
    staged_paths["home"] = str(staging_dir)
    staged_opts["paths"] = staged_paths
    staged_opts["outtmpl"] = str(template_path)
    try:
        yield staged_opts, output_dir, staging_dir
    finally:
        with contextlib.suppress(BaseException):
            _remove_staging_path(staging_dir)


def _commit_staged_artifact(
    staging_dir: Path,
    output_dir: Path,
    video_id: str,
    *,
    media_type: str,
) -> Path | None:
    """Validate one staged artifact and atomically move it into the output root."""
    artifact = _find_completed_artifact(
        staging_dir,
        video_id,
        media_type=media_type,
    )
    if artifact is None:
        return None
    final = safe_output_path(output_dir, artifact.name)
    if final.exists():
        if final.is_symlink() or not final.is_file():
            raise ValueError(f"output artifact path is not a regular file: {final}")
        if not is_completed_artifact(final, media_type=media_type):
            final.unlink()
    artifact.replace(final)
    if not is_completed_artifact(final, media_type=media_type):
        with contextlib.suppress(OSError):
            final.unlink()
        return None
    if final.suffix.casefold() == ".webm":
        marker = safe_output_path(output_dir, f".{video_id}.webm-type")
        marker.write_text(f"{media_type}\n", encoding="ascii")
        marker.chmod(0o600)
    return final


async def _retry_operation(
    operation: Any,
    config: ScraperConfig,
    label: str,
    *,
    request_limiter: RequestLimiter | None = None,
) -> Any:
    """Run one transport operation; retries belong to the post workflow."""
    del config, label
    if request_limiter is not None:
        await request_limiter.acquire()
    operation_task = asyncio.ensure_future(operation())
    try:
        return await asyncio.shield(operation_task)
    except asyncio.CancelledError:
        # Cancelling the asyncio wrapper cannot stop a yt-dlp thread. Wait for
        # it to finish before propagating cancellation so it cannot write after
        # the caller has cleaned up the output tree.
        with contextlib.suppress(BaseException):
            await asyncio.shield(operation_task)
        raise


def _is_retryable_download_error(exc: Exception) -> bool:
    """Return whether a download error belongs to the transient class."""
    return _classify_download_error(exc) is _FailureKind.TRANSIENT


def _classify_download_error(exc: Exception) -> _FailureKind:
    """Separate permanent remote, transient network, and local failures."""
    if isinstance(exc, (DownloadArtifactError, MetadataMismatchError)):
        return _FailureKind.LOCAL

    if isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return _FailureKind.LOCAL

    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {408, 425, 429} or exc.code >= 500:
            return _FailureKind.TRANSIENT
        return _FailureKind.PERMANENT

    message = str(exc).lower()
    permanent = (
        "404",
        "not found",
        "does not exist",
        "unsupported url",
        "private video",
        "login required",
        "copyright",
        "forbidden",
        "invalid format",
        "not available",
    )
    if any(phrase in message for phrase in permanent):
        return _FailureKind.PERMANENT

    local = (
        "permission denied",
        "read-only file system",
        "no space left on device",
        "is a directory",
        "invalid argument",
    )
    if any(phrase in message for phrase in local):
        return _FailureKind.LOCAL

    transient = (
        "429",
        "rate limit",
        "too many requests",
        "temporarily",
        "timeout",
        "timed out",
        "network",
        "connection",
        "500",
        "502",
        "503",
        "504",
        "server error",
    )
    if isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return _FailureKind.TRANSIENT
    if isinstance(exc, OSError):
        return (
            _FailureKind.TRANSIENT
            if any(phrase in message for phrase in transient)
            else _FailureKind.LOCAL
        )
    if any(phrase in message for phrase in transient):
        return _FailureKind.TRANSIENT
    return _FailureKind.PERMANENT


def _failed_download(exc: Exception) -> _DownloadFailure:
    """Convert an exception to a classified internal workflow result."""
    return _DownloadFailure(exc, _classify_download_error(exc))


def _retry_after_from_exception(exc: Exception) -> float | None:
    """Read a Retry-After hint when an HTTP error exposes one."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    return parse_retry_after(str(value)) if value is not None else None


def _build_extract_opts(config: ScraperConfig) -> dict[str, Any]:
    """Construct yt-dlp options for metadata-only extraction."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": False,
        "logger": _YTDLP_LOGGER,
        "socket_timeout": config.request_timeout,
        "sleep_interval_requests": config.request_delay,
        "impersonate": _IMPERSONATE_CHROME,
        "extractor_retries": 0,
    }
    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)
    return opts


def _has_video_formats(info: dict[str, Any]) -> bool:
    """Return ``True`` if the post has at least one format with a video codec."""
    raw_formats = info.get("formats", [])
    formats = raw_formats if isinstance(raw_formats, list) else []
    if not formats:
        # No explicit format list — check top-level vcodec.
        vcodec = str(info.get("vcodec", "none"))
        return vcodec != "none"
    return any(
        isinstance(fmt, dict) and str(fmt.get("vcodec", "none")) != "none"
        for fmt in formats
    )


def _has_audio_formats(info: dict[str, Any]) -> bool:
    """Return whether the post has at least one format with an audio codec."""
    raw_formats = info.get("formats", [])
    formats = raw_formats if isinstance(raw_formats, list) else []
    if not formats:
        return str(info.get("acodec", "none")) != "none"
    return any(
        isinstance(fmt, dict) and str(fmt.get("acodec", "none")) != "none"
        for fmt in formats
    )


def _info_is_photo_post(info: dict[str, Any]) -> bool:
    """Detect explicit slideshow metadata independently of the post URL."""
    for key in (
        "imagePost",
        "image_post",
        "imagePostInfo",
        "image_post_info",
        "imageList",
        "image_list",
        "displayImages",
        "display_images",
    ):
        image_post = info.get(key)
        if isinstance(image_post, dict):
            for image_key in ("images", "imageList", "displayImages", "display_images"):
                images = image_post.get(image_key)
                if isinstance(images, list) and images:
                    return True
        elif isinstance(image_post, list) and image_post:
            return True
    for key in ("itemType", "item_type", "postType", "post_type", "type"):
        value = info.get(key)
        if isinstance(value, str) and value.casefold() in {
            "photo",
            "image",
            "imagepost",
            "photo_post",
        }:
            return True
    return False


def _requires_ffmpeg(info: dict[str, Any]) -> bool:
    """Return whether available formats require a video/audio merge."""
    raw_formats = info.get("formats", [])
    formats = raw_formats if isinstance(raw_formats, list) else []
    if not formats:
        return (
            str(info.get("vcodec", "none")) != "none"
            and str(info.get("acodec", "none")) == "none"
        )
    valid_formats = [fmt for fmt in formats if isinstance(fmt, dict)]
    has_video = any(
        str(fmt.get("vcodec", "none")) != "none" for fmt in valid_formats
    )
    has_progressive = any(
        str(fmt.get("vcodec", "none")) != "none"
        and str(fmt.get("acodec", "none")) != "none"
        for fmt in valid_formats
    )
    return has_video and not has_progressive


def _extract_image_urls(
    info: dict[str, Any],
    *,
    is_photo_post: bool = False,
) -> list[str]:
    """Extract slideshow image URLs from yt-dlp metadata.

    yt-dlp's TikTok extractor stores slideshow images in several
    possible locations depending on the version:

    1. ``thumbnails`` list where each entry has a ``url`` and is an
       actual content image (not just a thumbnail).
    2. Top-level ``url`` pointing to a single image.
    3. Nested ``imagePost.images`` structures.

    Returns:
        A deduplicated, ordered list of image URLs.  Empty if no images
        are found.
    """
    urls: list[str] = []
    seen: set[str] = set()

    def _add(url: str, *, require_image_suffix: bool = True) -> None:
        try:
            parsed_url = urlsplit(url)
        except ValueError:
            return
        if (
            isinstance(url, str)
            and url
            and parsed_url.scheme.casefold() == "https"
            and parsed_url.hostname is not None
            and (not require_image_suffix or _looks_like_image_url(url))
            and url not in seen
        ):
            seen.add(url)
            urls.append(url)

    # Prefer semantic image-post data. It identifies content images rather
    # than generic cover thumbnails and may use extensionless CDN URLs.
    image_post = info.get(
        "imagePost",
        info.get("image_post", info.get("imagePostInfo", {})),
    )
    if not image_post:
        image_post = info.get(
            "imageList",
            info.get("image_list", info.get("displayImages", [])),
        )
    if isinstance(image_post, list):
        image_post = {"images": image_post}
    if isinstance(image_post, dict):
        images: list[Any] = []
        for key in ("images", "displayImages", "display_images", "imageList"):
            values = image_post.get(key)
            if isinstance(values, list):
                images.extend(values)
        for img in images:
            if not isinstance(img, dict):
                continue
            url_info = img.get("imageURL", img.get("image_url", img))
            if not isinstance(url_info, dict):
                continue
            url_list = url_info.get("urlList", url_info.get("url_list", []))
            if isinstance(url_list, list):
                candidates = [url for url in url_list if isinstance(url, str)]
                if candidates:
                    _add(candidates[0], require_image_suffix=False)
        if urls:
            return urls

    # Strategy 1: image-like formats exposed without a filename extension.
    if is_photo_post:
        raw_formats = info.get("formats", [])
        formats = raw_formats if isinstance(raw_formats, list) else []
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            if str(fmt.get("vcodec", "none")) != "none":
                continue
            if str(fmt.get("acodec", "none")) != "none":
                continue
            format_url = fmt.get("url")
            if isinstance(format_url, str):
                _add(format_url, require_image_suffix=False)

    # Strategy 2: entries that are images (for playlist-style extraction).
    raw_entries = info.get("entries", [])
    entries = raw_entries if isinstance(raw_entries, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_url = entry.get("url", "")
        if isinstance(entry_url, str):
            _add(entry_url, require_image_suffix=not is_photo_post)

    # Strategy 3: explicit image URL collections used by some extractor
    # versions. These are only trusted when the post is known to be a photo.
    if is_photo_post:
        for key in ("image_urls", "imageUrls", "images"):
            values = info.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str):
                    _add(value, require_image_suffix=False)
                elif isinstance(value, dict):
                    url = value.get("url")
                    if isinstance(url, str):
                        _add(url, require_image_suffix=False)

    # Strategy 4: thumbnails are a compatibility fallback for older yt-dlp
    # payloads, but never infer a slideshow from generic video thumbnails.
    if is_photo_post:
        raw_thumbnails = info.get("thumbnails", [])
        thumbnails = raw_thumbnails if isinstance(raw_thumbnails, list) else []
        for thumb in thumbnails:
            if not isinstance(thumb, dict):
                continue
            thumb_url = thumb.get("url", "")
            if isinstance(thumb_url, str):
                _add(thumb_url)

    # Strategy 5: top-level URL when it is an image.
    top_url = info.get("url", "")
    if is_photo_post and isinstance(top_url, str):
        _add(top_url, require_image_suffix=False)

    return urls


def _looks_like_image_url(url: str) -> bool:
    """Heuristic check for whether a URL points to an image."""
    if not isinstance(url, str) or not url:
        return False
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return False
    lower = parsed.path.lower()
    return any(
        lower.endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif")
    )


def _validate_public_image_url(url: str) -> None:
    """Reject non-HTTPS, local, or unresolved image destinations."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("image URL must be valid HTTPS") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ValueError("image URL must use HTTPS without credentials or a custom port")

    addresses = _resolve_public_addresses(hostname, port or 443)
    if not addresses:
        raise ValueError("image URL hostname could not be resolved")


def _resolve_public_addresses(hostname: str, port: int) -> tuple[str, ...]:
    """Resolve a host and return only addresses safe for a public request."""
    try:
        address_infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("image URL hostname could not be resolved") from exc
    addresses: list[str] = []
    for address_info in address_infos:
        address_value = address_info[4][0]
        try:
            address = ipaddress.ip_address(address_value)
        except ValueError as exc:
            raise ValueError("image URL resolved to an invalid address") from exc
        if not address.is_global:
            raise ValueError("image URL must not target a private address")
        addresses.append(str(address_value))
    return tuple(addresses)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a previously validated address while preserving TLS SNI."""

    def __init__(self, host: str, *, resolved_address: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._resolved_address = resolved_address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._resolved_address, self.port),
            self.timeout,
            getattr(self, "source_address", None),
        )
        tunnel_host = getattr(self, "_tunnel_host", None)
        if tunnel_host:
            self._tunnel()  # type: ignore[attr-defined]
        context = self._context  # type: ignore[attr-defined]
        self.sock = context.wrap_socket(
            self.sock,
            server_hostname=tunnel_host or self.host,
        )


class _SafeHTTPSHandler(urllib.request.HTTPSHandler):
    """Use a validated, pinned address for each HTTPS request."""

    def https_open(self, req: urllib.request.Request) -> Any:
        _validate_public_image_url(req.full_url)
        parsed = urlsplit(req.full_url)
        hostname = parsed.hostname
        if hostname is None:  # pragma: no cover - validation above enforces this
            raise ValueError("image URL hostname is missing")
        addresses = _resolve_public_addresses(hostname, parsed.port or 443)
        return self.do_open(
            functools.partial(
                _PinnedHTTPSConnection,
                resolved_address=addresses[0],
            ),
            req,
            context=getattr(self, "_context", None),
        )


class _SafeImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate redirect destinations and re-scope cookies on every hop."""

    def __init__(self, cookies_file: Path | None) -> None:
        super().__init__()
        self._cookies_file = cookies_file

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target_url = urljoin(req.full_url, newurl)
        _validate_public_image_url(target_url)
        redirected = super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            target_url,
        )
        if redirected is None:
            return None
        redirected.remove_header("Cookie")
        redirected.unredirected_hdrs.pop("Cookie", None)
        cookie_pairs = read_netscape_cookie_pairs(
            self._cookies_file,
            url=target_url,
        )
        if cookie_pairs:
            redirected.add_header(
                "Cookie",
                "; ".join(
                    f"{name}={value}" for name, value in cookie_pairs.items()
                ),
            )
        return redirected


def _open_image_url(
    request: urllib.request.Request,
    *,
    cookies_file: Path | None,
    timeout: float,
) -> Any:
    """Open an image URL through the redirect-safe opener."""
    _validate_public_image_url(request.full_url)
    opener = urllib.request.build_opener(
        # A proxy would make the connection target differ from the address we
        # pinned, so this security-sensitive path connects directly.
        urllib.request.ProxyHandler({}),
        _SafeImageRedirectHandler(cookies_file),
        _SafeHTTPSHandler(),
    )
    return opener.open(request, timeout=timeout)


def _set_response_read_timeout(response: Any, timeout: float) -> None:
    """Apply the remaining deadline to urllib's underlying socket when present."""
    candidates = [response]
    file_pointer = getattr(response, "fp", None)
    if file_pointer is not None:
        candidates.append(file_pointer)
        raw = getattr(file_pointer, "raw", None)
        if raw is not None:
            candidates.append(raw)
            socket_object = getattr(raw, "_sock", None)
            if socket_object is not None:
                candidates.append(socket_object)
    for candidate in candidates:
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            with contextlib.suppress(OSError):
                setter(max(0.001, timeout))
            return


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------


async def _download_images(
    image_urls: list[str],
    output_dir: Path,
    video_id: str,
    image_semaphore: asyncio.Semaphore,
    *,
    retries: int = 3,
    retry_backoff_base: float = 1.0,
    retry_backoff_max: float = 60.0,
    request_limiter: RequestLimiter | None = None,
    cookies_file: Path | None = None,
    timeout: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
) -> None:
    """Download slideshow images into ``output_dir/<video_id>/``.

    Images are first written to a non-numeric staging directory.  The
    directory is renamed only after every image succeeds so resume detection
    never mistakes a partial slideshow for a completed one.

    Args:
        image_urls: URLs of the images to download.
        output_dir: Base output directory.
        video_id: The TikTok post ID (used as subdirectory name).
        image_semaphore: Session-wide limit for concurrent image downloads.
        retries: Number of retry attempts per image on transient failures.

    Raises:
        Exception: If any image download fails after all retries.
    """
    if not is_valid_video_id(video_id):
        raise ValueError("video_id must contain 15 to 25 decimal digits")
    validate_output_dir(output_dir, create=True)
    photo_dir = safe_output_path(output_dir, video_id)
    staging_dir = safe_output_path(output_dir, f".{video_id}.partial")

    _remove_staging_path(staging_dir)
    if photo_dir.exists():
        if is_completed_slideshow_artifact(photo_dir):
            raise DownloadArtifactError(f"slideshow already exists: {video_id}")
        if photo_dir.is_dir():
            shutil.rmtree(photo_dir)
        else:
            raise OSError(f"output path is not a slideshow directory: {photo_dir}")
    staging_dir.mkdir(parents=False, exist_ok=False, mode=0o700)

    async def _download_one(idx: int, url: str) -> None:
        ext = _image_extension(url)
        dest = safe_output_path(staging_dir, f"image_{idx}{ext}")
        async with image_semaphore:
            if request_limiter is not None:
                await request_limiter.acquire()
            download_task = asyncio.create_task(
                asyncio.to_thread(
                    _sync_download_file,
                    url,
                    dest,
                    retries=retries,
                    retry_backoff_base=retry_backoff_base,
                    retry_backoff_max=retry_backoff_max,
                    cookies_file=cookies_file,
                    timeout=timeout,
                    max_bytes=max_bytes,
                )
            )
            try:
                await asyncio.shield(download_task)
            except asyncio.CancelledError:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(download_task)
                raise
        logger.debug("Saved %s", dest)

    download_tasks = [
        asyncio.create_task(_download_one(idx, url))
        for idx, url in enumerate(image_urls, start=1)
    ]
    try:
        outcomes = await asyncio.gather(*download_tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

        expected_files = {
            f"image_{idx}{_image_extension(url)}"
            for idx, url in enumerate(image_urls, start=1)
        }
        manifest_path = safe_output_path(staging_dir, ".tiktok-complete.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "video_id": video_id,
                    "files": sorted(expected_files),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)
        children = [
            child
            for child in staging_dir.iterdir()
            if child.name != ".tiktok-complete.json"
        ]
        if {child.name for child in children} != expected_files or not (
            children
            and all(is_completed_artifact(child, media_type="image") for child in children)
        ):
            raise DownloadArtifactError(
                f"slideshow did not produce all {len(image_urls)} images"
            )

        staging_dir.replace(photo_dir)
        if not is_completed_slideshow_artifact(photo_dir):
            raise DownloadArtifactError("committed slideshow failed validation")
    except asyncio.CancelledError:
        for task in download_tasks:
            task.cancel()
        await asyncio.gather(*download_tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            _remove_staging_path(staging_dir)
        raise
    except BaseException:
        with contextlib.suppress(BaseException):
            _remove_staging_path(staging_dir)
        raise


def _remove_staging_path(path: Path) -> None:
    """Remove a downloader-owned staging path without following symlinks."""
    if path.is_symlink():
        raise ValueError(f"symlinked staging path is not allowed: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _sync_download_file(
    url: str,
    dest: Path,
    *,
    retries: int = 3,
    retry_backoff_base: float = 1.0,
    retry_backoff_max: float = 60.0,
    cookies_file: Path | None = None,
    timeout: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
) -> None:
    """Download a single file via urllib with retry logic (runs in an executor).

    Retries on HTTP 429 (rate-limit) and 5xx server errors with
    exponential backoff.  Non-retryable HTTP errors and other
    exceptions are raised immediately.

    Args:
        url: The URL to download.
        dest: Destination file path.
        retries: Maximum number of retry attempts.

    Raises:
        urllib.error.URLError: If the download fails after all retries.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    dest = safe_output_path(dest.parent, dest.name)
    temp_dest = safe_output_path(dest.parent, f"{dest.name}.part")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.tiktok.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    cookie_pairs = read_netscape_cookie_pairs(cookies_file, url=url)
    if cookie_pairs:
        headers["Cookie"] = "; ".join(
            f"{name}={value}" for name, value in cookie_pairs.items()
        )
    req = urllib.request.Request(url, headers=headers)

    last_exc: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        if temp_dest.is_symlink():
            raise ValueError(f"symlinked temporary path is not allowed: {temp_dest}")
        if temp_dest.exists():
            temp_dest.unlink()
        try:
            deadline = time.monotonic() + timeout
            with _open_image_url(
                req,
                cookies_file=cookies_file,
                timeout=timeout,
            ) as resp:
                response_headers = getattr(resp, "headers", None)
                content_length = (
                    response_headers.get("Content-Length")
                    if response_headers is not None
                    else None
                )
                if isinstance(content_length, str):
                    try:
                        if int(content_length) > max_bytes:
                            raise DownloadArtifactError(
                                f"image exceeds the {max_bytes}-byte limit"
                            )
                    except ValueError:
                        pass
                get_content_type = getattr(response_headers, "get_content_type", None)
                content_type = (
                    get_content_type() if callable(get_content_type) else ""
                )
                if isinstance(content_type, str) and content_type not in {
                    "",
                    "application/octet-stream",
                } and not content_type.casefold().startswith("image/"):
                    raise DownloadArtifactError(
                        f"image response has unexpected content type {content_type}"
                    )
                opened_descriptor = os.open(
                    temp_dest,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW,
                    0o600,
                )
                file_descriptor: int | None = opened_descriptor
                try:
                    os.fchmod(opened_descriptor, 0o600)
                    with os.fdopen(opened_descriptor, "wb") as output:
                        file_descriptor = None
                        total_bytes = 0
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("image download exceeded its deadline")
                            _set_response_read_timeout(resp, remaining)
                            chunk = resp.read(
                                min(_DOWNLOAD_CHUNK_SIZE, max_bytes - total_bytes + 1)
                            )
                            if not chunk:
                                break
                            total_bytes += len(chunk)
                            if total_bytes > max_bytes:
                                raise DownloadArtifactError(
                                    f"image exceeds the {max_bytes}-byte limit"
                                )
                            output.write(chunk)
                finally:
                    if file_descriptor is not None:
                        with contextlib.suppress(OSError):
                            os.close(file_descriptor)
                temp_dest.replace(dest)
                if not is_completed_artifact(dest, media_type="image"):
                    dest.unlink()
                    raise DownloadArtifactError(f"invalid image artifact: {url}")
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in {408, 425, 429} or exc.code >= 500:
                if attempt >= attempts:
                    raise urllib.error.URLError(
                        f"Failed to download {url} after {attempts} attempts"
                    ) from exc
                retry_after = parse_retry_after(
                    exc.headers.get("Retry-After") if exc.headers is not None else None
                )
                delay = bounded_backoff(
                    attempt,
                    base=retry_backoff_base,
                    cap=retry_backoff_max,
                    retry_after=retry_after,
                )
                logger.warning(
                    "HTTP %d downloading %s (attempt %d/%d), retrying in %.1fs...",
                    exc.code,
                    redact_log_value(url),
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable HTTP error (4xx other than 429).
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt >= attempts:
                raise urllib.error.URLError(
                    f"Failed to download {url} after {attempts} attempts"
                ) from exc
            delay = bounded_backoff(
                attempt,
                base=retry_backoff_base,
                cap=retry_backoff_max,
            )
            logger.warning(
                "Network error downloading %s (attempt %d/%d): %s, retrying in %.1fs...",
                redact_log_value(url),
                attempt,
                attempts,
                redact_log_value(exc),
                delay,
            )
            time.sleep(delay)
            continue
        except OSError as exc:
            if _classify_download_error(exc) is not _FailureKind.TRANSIENT:
                # Filesystem failures are local and should not be retried.
                raise
            last_exc = exc
            if attempt >= attempts:
                raise urllib.error.URLError(
                    f"Failed to download {url} after {attempts} attempts"
                ) from exc
            delay = bounded_backoff(
                attempt,
                base=retry_backoff_base,
                cap=retry_backoff_max,
            )
            logger.warning(
                "Network error downloading %s (attempt %d/%d): %s, retrying in %.1fs...",
                redact_log_value(url),
                attempt,
                attempts,
                redact_log_value(exc),
                delay,
            )
            time.sleep(delay)
            continue
        finally:
            if temp_dest.exists() and not temp_dest.is_symlink():
                try:
                    temp_dest.unlink()
                except OSError:
                    logger.debug("Could not remove temporary image %s", temp_dest)

    # Exhausted all retries.
    msg = f"Failed to download {url} after {attempts} attempts"
    raise urllib.error.URLError(msg) from last_exc


def _image_extension(url: str) -> str:
    """Derive a file extension from an image URL.  Defaults to ``.jpg``."""
    path = url.split("?")[0].split("#")[0]
    for ext in (".png", ".webp", ".avif", ".gif", ".jpeg", ".jpg"):
        if path.lower().endswith(ext):
            return ext
    return ".jpg"


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------


def _sync_download_info(
    info: dict[str, Any],
    opts: dict[str, Any],
    expected_id: str | None = None,
    media_type: str | None = None,
) -> bool:
    """Download pre-extracted yt-dlp metadata without resolving its URL again."""
    info_id = str(info.get("id", ""))
    artifact_id = expected_id or info_id
    if expected_id is not None and not is_valid_video_id(expected_id):
        return False
    if expected_id is not None and (
        info_id != expected_id or not is_valid_video_id(info_id)
    ):
        return False

    artifact_type = media_type or ("video" if _has_video_formats(info) else "audio")
    with _staged_ydl_opts(opts, artifact_id) as (
        staged_opts,
        output_dir,
        staging_dir,
    ):
        with _isolated_ydl_opts(staged_opts) as safe_opts, yt_dlp.YoutubeDL(
            safe_opts
        ) as ydl:
            ydl.process_ie_result(info, download=True)
        if output_dir is None or staging_dir is None:
            return False
        return (
            _commit_staged_artifact(
                staging_dir,
                output_dir,
                artifact_id,
                media_type=artifact_type,
            )
            is not None
        )


def _find_completed_artifact(
    output_dir: Path,
    video_id: str,
    *,
    media_type: str | None = None,
) -> Path | None:
    """Find a strict media artifact for a post ID in one output directory."""
    if not is_valid_video_id(video_id):
        return None
    root = validate_output_dir(output_dir)
    if not root.exists():
        return None
    prefix = f"{video_id}."
    for path in root.iterdir():
        if not path.name.startswith(prefix):
            continue
        if is_completed_artifact(path, media_type=media_type):
            return path
    return None


def _sync_download_embed(
    video_id: str,
    opts: dict[str, Any],
) -> dict[str, Any] | None:
    """Download the first direct video exposed by TikTok's embed page."""
    if not is_valid_video_id(video_id):
        raise ValueError("video_id must contain 15 to 25 decimal digits")
    embed_url = f"https://www.tiktok.com/embed/v2/{video_id}"
    ydl_opts = dict(opts)
    expected_author = ydl_opts.pop("_expected_author", None)
    with _staged_ydl_opts(ydl_opts, video_id) as (
        staged_opts,
        output_dir,
        staging_dir,
    ):
        with _isolated_ydl_opts(staged_opts) as safe_opts, yt_dlp.YoutubeDL(
            safe_opts
        ) as ydl:
            result = ydl.extract_info(embed_url, download=True)
        if result is None:
            raise yt_dlp.utils.DownloadError("TikTok embed returned no metadata")
        result_info = dict(result) if isinstance(result, dict) else None
        if result_info is not None:
            result_id = str(result_info.get("id", ""))
            if result_id != video_id or not is_valid_video_id(result_id):
                raise MetadataMismatchError(
                    "TikTok embed metadata does not identify the requested post"
                )
            if isinstance(expected_author, str):
                expected_video = VideoInfo(
                    video_id=video_id,
                    url=(
                        f"https://www.tiktok.com/@{expected_author}/"
                        f"video/{video_id}"
                    ),
                )
                if not _metadata_matches_video(
                    result_info,
                    expected_video,
                    require_identity=False,
                ):
                    raise MetadataMismatchError(
                        "TikTok embed metadata does not identify the requested creator"
                    )
        elif output_dir is not None:
            raise MetadataMismatchError("TikTok embed returned no structured metadata")
        if (
            output_dir is not None
            and staging_dir is not None
            and _commit_staged_artifact(
                staging_dir,
                output_dir,
                video_id,
                media_type="video",
            )
            is None
        ):
            raise DownloadArtifactError(f"TikTok embed did not create {video_id}.mp4")
        return result_info


def _build_ydl_opts(
    config: ScraperConfig,
    *,
    merge: bool = True,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Construct the yt-dlp options dictionary."""
    output_dir = validate_output_dir(config.output_dir)
    if video_id is not None:
        if not is_valid_video_id(video_id):
            raise ValueError("video_id must contain 15 to 25 decimal digits")
        safe_output_path(output_dir, f"{video_id}.%(ext)s")
        outtmpl = f"{video_id}.%(ext)s"
    else:
        outtmpl = "%(id)s.%(ext)s"

    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "paths": {"home": str(output_dir)},
        "format": (
            "bestvideo*+bestaudio/best"
            if merge
            else "best[vcodec!=none][acodec!=none]/best"
        ),
        "quiet": True,
        "no_warnings": False,
        "logger": _YTDLP_LOGGER,
        "socket_timeout": config.request_timeout,
        # The post-level retry loop is the single retry budget owner.
        "retries": 0,
        "extractor_retries": 0,
        "fragment_retries": 0,
        "noprogress": True,
        # Conservative spacing complements the bounded wrapper backoff.
        "sleep_interval_requests": config.request_delay,
        "sleep_interval": config.request_delay,
        "max_sleep_interval": 2 * config.request_delay,
        "impersonate": _IMPERSONATE_CHROME,
    }

    if merge:
        opts["merge_output_format"] = "mp4"

    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)

    return opts


def _build_embed_ydl_opts(
    config: ScraperConfig,
    video_id: str,
) -> dict[str, Any]:
    """Construct yt-dlp options for the TikTok embed fallback."""
    if not is_valid_video_id(video_id):
        raise ValueError("video_id must contain 15 to 25 decimal digits")
    output_dir = validate_output_dir(config.output_dir)
    safe_output_path(output_dir, f"{video_id}.mp4")
    opts: dict[str, Any] = {
        "outtmpl": f"{video_id}.mp4",
        "paths": {"home": str(output_dir)},
        "quiet": True,
        "no_warnings": False,
        "logger": _YTDLP_LOGGER,
        "socket_timeout": config.request_timeout,
        # The post-level retry loop is the single retry budget owner.
        "retries": 0,
        "extractor_retries": 0,
        "fragment_retries": 0,
        "noprogress": True,
        # Start fresh instead of resuming a partial primary-download file.
        "continuedl": False,
        "overwrites": True,
        # The embed page exposes the same source twice; only download it once.
        "playlistend": 1,
        "impersonate": _IMPERSONATE_CHROME,
        "sleep_interval_requests": config.request_delay,
        "sleep_interval": config.request_delay,
        "max_sleep_interval": 2 * config.request_delay,
    }

    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)

    return opts
