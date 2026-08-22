"""Video and photo download via yt-dlp with concurrency control."""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yt_dlp

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.models import ScrapeResult, VideoInfo
from tiktok_scraper.utils import (
    bounded_backoff,
    check_ffmpeg,
    cleanup_temporary_artifacts,
    get_downloaded_ids,
    is_completed_artifact,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_IMAGE_DOWNLOADS = 3
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class _DownloadOutcome(enum.Enum):
    """Result of a single download attempt."""

    VIDEO = "video"
    PHOTOS = "photos"
    SKIPPED = "skipped"
    SKIPPED_AUDIO_ONLY = "skipped_audio_only"
    FAILED = "failed"


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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_temporary_artifacts(config.output_dir)

    existing_ids = get_downloaded_ids(config.output_dir)
    result = ScrapeResult(
        username=config.username,
        total_found=len(videos),
    )

    to_download: list[VideoInfo] = []
    for video in videos:
        if video.video_id in existing_ids:
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
    queue: asyncio.Queue[tuple[int, VideoInfo] | None] = asyncio.Queue()
    outcomes: list[_DownloadOutcome | Exception | None] = [None] * len(to_download)

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
                    outcomes[outcome_index] = await _download_single(
                        video,
                        config,
                        outcome_index + 1,
                        len(to_download),
                        image_semaphore,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    outcomes[outcome_index] = exc
            finally:
                queue.task_done()

    worker_count = min(config.max_concurrent_downloads, len(to_download))
    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    await queue.join()

    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    for video, outcome in zip(to_download, outcomes, strict=True):
        if isinstance(outcome, Exception):
            result.failed += 1
            result.failed_videos.append(video.video_id)
            logger.error("Download failed for %s: %s", video.video_id, outcome)
        elif outcome is _DownloadOutcome.VIDEO:
            result.downloaded += 1
        elif outcome is _DownloadOutcome.PHOTOS:
            result.photos_downloaded += 1
        elif outcome is _DownloadOutcome.SKIPPED:
            result.skipped += 1
        elif outcome is _DownloadOutcome.SKIPPED_AUDIO_ONLY:
            result.skipped_audio_only += 1
        else:
            result.failed += 1
            result.failed_videos.append(video.video_id)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _download_single(
    video: VideoInfo,
    config: ScraperConfig,
    index: int,
    total: int,
    image_semaphore: asyncio.Semaphore,
) -> _DownloadOutcome:
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
        )
    except Exception as exc:
        logger.error(
            "[%d/%d] Failed to extract info for %s: %s",
            index,
            total,
            video.video_id,
            exc,
        )
        return _DownloadOutcome.FAILED

    if info is None:
        logger.warning(
            "[%d/%d] No metadata returned for %s.",
            index,
            total,
            video.video_id,
        )
        return _DownloadOutcome.FAILED

    has_video = _has_video_formats(info)

    if has_video:
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
            return _DownloadOutcome.FAILED
        ydl_opts = _build_ydl_opts(config, merge=ffmpeg_available)
        try:
            artifact_valid = await _retry_operation(
                lambda: loop.run_in_executor(
                    None,
                    functools.partial(_sync_download_info, info, ydl_opts),
                ),
                config,
                f"video {video.video_id}",
            )
            if artifact_valid is False or artifact_valid is None:
                raise DownloadArtifactError(video.video_id)
            logger.info("[%d/%d] Completed video %s.", index, total, video.video_id)
            return _DownloadOutcome.VIDEO
        except Exception as exc:
            logger.error(
                "[%d/%d] Failed video %s: %s",
                index,
                total,
                video.video_id,
                exc,
            )
            return _DownloadOutcome.FAILED

    # No video stream — check for slideshow images.
    image_urls = _extract_image_urls(info)

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
                retries=config.download_retries,
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
                exc,
            )
            return _DownloadOutcome.FAILED

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
    ydl_opts = _build_ydl_opts(config, merge=False)
    try:
        artifact_valid = await _retry_operation(
            lambda: loop.run_in_executor(
                None,
                functools.partial(_sync_download_info, info, ydl_opts),
            ),
            config,
            f"audio {video.video_id}",
        )
        if artifact_valid is False or artifact_valid is None:
            raise DownloadArtifactError(video.video_id)
        logger.info("[%d/%d] Completed audio %s.", index, total, video.video_id)
        return _DownloadOutcome.VIDEO
    except Exception as exc:
        logger.error(
            "[%d/%d] Failed audio %s: %s",
            index,
            total,
            video.video_id,
            exc,
        )
        return _DownloadOutcome.FAILED


# ---------------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------------


def _sync_extract_info(
    url: str,
    opts: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract post metadata without downloading (runs in an executor)."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.extract_info(url, download=False)
        if result is None:
            return None
        return dict(result)


class DownloadArtifactError(RuntimeError):
    """Raised when yt-dlp returns without producing a completed artifact."""


async def _retry_operation(operation: Any, config: ScraperConfig, label: str) -> Any:
    """Retry transient yt-dlp work with bounded exponential backoff."""
    attempts = max(1, config.download_retries)
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if not _is_retryable_download_error(exc) or attempt >= attempts:
                raise
            delay = bounded_backoff(
                attempt,
                base=config.retry_backoff_base,
                cap=config.retry_backoff_max,
                retry_after=_retry_after_from_exception(exc),
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


def _is_retryable_download_error(exc: Exception) -> bool:
    """Classify yt-dlp errors conservatively for retry purposes."""
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
        "permission denied",
        "invalid format",
    )
    if any(phrase in message for phrase in permanent):
        return False
    if isinstance(exc, DownloadArtifactError):
        return False
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    transient = (
        "429",
        "rate limit",
        "too many requests",
        "temporarily",
        "timeout",
        "timed out",
        "network",
        "connection",
        "502",
        "503",
        "504",
    )
    return any(phrase in message for phrase in transient)


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
        "no_warnings": True,
        "sleep_interval_requests": config.request_delay,
    }
    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)
    return opts


def _has_video_formats(info: dict[str, Any]) -> bool:
    """Return ``True`` if the post has at least one format with a video codec."""
    formats: list[dict[str, Any]] = info.get("formats", [])
    if not formats:
        # No explicit format list — check top-level vcodec.
        vcodec = str(info.get("vcodec", "none"))
        return vcodec != "none"
    return any(str(fmt.get("vcodec", "none")) != "none" for fmt in formats)


def _requires_ffmpeg(info: dict[str, Any]) -> bool:
    """Return whether available formats require a video/audio merge."""
    formats: list[dict[str, Any]] = info.get("formats", [])
    if not formats:
        return (
            str(info.get("vcodec", "none")) != "none"
            and str(info.get("acodec", "none")) == "none"
        )
    has_video = any(str(fmt.get("vcodec", "none")) != "none" for fmt in formats)
    has_progressive = any(
        str(fmt.get("vcodec", "none")) != "none"
        and str(fmt.get("acodec", "none")) != "none"
        for fmt in formats
    )
    return has_video and not has_progressive


def _extract_image_urls(info: dict[str, Any]) -> list[str]:
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

    def _add(url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    # Strategy 1: entries that are images (for playlist-style extraction).
    entries: list[dict[str, Any]] = info.get("entries", [])
    for entry in entries:
        if entry is None:
            continue
        entry_url = entry.get("url", "")
        if _looks_like_image_url(entry_url):
            _add(entry_url)

    # Strategy 2: thumbnails list — yt-dlp TikTok slideshow support puts
    # full-size images here in newer versions.
    thumbnails: list[dict[str, Any]] = info.get("thumbnails", [])
    for thumb in thumbnails:
        thumb_url = thumb.get("url", "")
        if _looks_like_image_url(thumb_url):
            _add(thumb_url)

    # Strategy 3: top-level URL when it is an image.
    top_url = info.get("url", "")
    if _looks_like_image_url(top_url):
        _add(top_url)

    # Strategy 4: nested imagePost structures (raw TikTok API data that
    # yt-dlp may expose via _api_data or similar fields).
    image_post = info.get("imagePost", {})
    if isinstance(image_post, dict):
        images = image_post.get("images", [])
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    url_info = img.get("imageURL", {})
                    if isinstance(url_info, dict):
                        url_list = url_info.get("urlList", [])
                        if isinstance(url_list, list) and url_list:
                            _add(str(url_list[0]))

    return urls


def _looks_like_image_url(url: str) -> bool:
    """Heuristic check for whether a URL points to an image."""
    if not url:
        return False
    lower = url.lower().split("?")[0]
    return any(
        lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif")
    )


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
    photo_dir = output_dir / video_id
    staging_dir = output_dir / f".{video_id}.partial"
    staging_dir.mkdir(parents=True, exist_ok=True)

    async def _download_one(idx: int, url: str) -> None:
        ext = _image_extension(url)
        dest = staging_dir / f"image_{idx}{ext}"
        async with image_semaphore:
            await asyncio.to_thread(
                _sync_download_file,
                url,
                dest,
                retries=retries,
            )
        logger.debug("Saved %s", dest)

    outcomes = await asyncio.gather(
        *(_download_one(idx, url) for idx, url in enumerate(image_urls, start=1)),
        return_exceptions=True,
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome

    staging_dir.replace(photo_dir)


def _sync_download_file(
    url: str,
    dest: Path,
    *,
    retries: int = 3,
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
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        },
    )

    last_exc: BaseException | None = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                temp_dest = dest.with_name(f"{dest.name}.part")
                with temp_dest.open("wb") as output:
                    while chunk := resp.read(_DOWNLOAD_CHUNK_SIZE):
                        output.write(chunk)
                temp_dest.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or exc.code >= 500:
                if attempt >= attempts:
                    raise urllib.error.URLError(
                        f"Failed to download {url} after {attempts} attempts"
                    ) from exc
                retry_after = parse_retry_after(
                    exc.headers.get("Retry-After") if exc.headers is not None else None
                )
                delay = bounded_backoff(
                    attempt,
                    base=1.0,
                    cap=60.0,
                    retry_after=retry_after,
                )
                logger.warning(
                    "HTTP %d downloading %s (attempt %d/%d), retrying in %.1fs...",
                    exc.code,
                    url,
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable HTTP error (4xx other than 429).
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt >= attempts:
                raise urllib.error.URLError(
                    f"Failed to download {url} after {attempts} attempts"
                ) from exc
            delay = bounded_backoff(attempt, base=1.0, cap=60.0)
            logger.warning(
                "Network error downloading %s (attempt %d/%d): %s, retrying in %.1fs...",
                url,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            continue

    # Exhausted all retries.
    msg = f"Failed to download {url} after {attempts} attempts"
    raise urllib.error.URLError(msg) from last_exc


def _image_extension(url: str) -> str:
    """Derive a file extension from an image URL.  Defaults to ``.jpg``."""
    path = url.split("?")[0].split("#")[0]
    for ext in (".png", ".webp", ".avif", ".jpeg", ".jpg"):
        if path.lower().endswith(ext):
            return ext
    return ".jpg"


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------


def _sync_download_info(
    info: dict[str, Any],
    opts: dict[str, Any],
) -> bool:
    """Download pre-extracted yt-dlp metadata without resolving its URL again."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.process_ie_result(info, download=True)
    outtmpl = opts.get("outtmpl")
    video_id = str(info.get("id", ""))
    if not isinstance(outtmpl, str) or not video_id:
        return False
    artifact = _find_completed_artifact(Path(outtmpl).parent, video_id)
    return artifact is not None


def _find_completed_artifact(output_dir: Path, video_id: str) -> Path | None:
    """Find a non-temporary, non-empty artifact for a post ID."""
    if not output_dir.exists():
        return None
    for path in output_dir.glob(f"{video_id}.*"):
        if is_completed_artifact(path) and path.stat().st_size > 0:
            return path
    return None


def _build_ydl_opts(
    config: ScraperConfig,
    *,
    merge: bool = True,
) -> dict[str, Any]:
    """Construct the yt-dlp options dictionary."""
    outtmpl = str(config.output_dir / "%(id)s.%(ext)s")

    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": (
            "bestvideo*+bestaudio/best"
            if merge
            else "best[vcodec!=none][acodec!=none]/best"
        ),
        "quiet": True,
        "no_warnings": True,
        "retries": config.download_retries,
        "fragment_retries": config.download_retries,
        "noprogress": True,
        # Conservative spacing complements the bounded wrapper backoff.
        "sleep_interval_requests": config.request_delay,
        "sleep_interval": config.request_delay,
        "max_sleep_interval": 2 * config.request_delay,
    }

    if merge:
        opts["merge_output_format"] = "mp4"

    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)

    return opts
