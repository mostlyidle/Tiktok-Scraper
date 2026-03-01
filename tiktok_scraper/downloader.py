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
from tiktok_scraper.utils import get_downloaded_ids

logger = logging.getLogger(__name__)


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
    ``config.max_concurrent_downloads`` at a time via an
    :class:`asyncio.Semaphore`.

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

    semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    async def _guarded_download(
        video: VideoInfo,
        index: int,
    ) -> _DownloadOutcome:
        async with semaphore:
            return await _download_single(video, config, index, len(to_download))

    tasks = [
        _guarded_download(video, idx) for idx, video in enumerate(to_download, start=1)
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    for video, outcome in zip(to_download, outcomes, strict=True):
        if isinstance(outcome, BaseException):
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
        info = await loop.run_in_executor(
            None,
            functools.partial(_sync_extract_info, video.url, extract_opts),
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
        ydl_opts = _build_ydl_opts(config)
        try:
            await loop.run_in_executor(
                None,
                functools.partial(_sync_download, video.url, ydl_opts),
            )
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
                loop,
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
    ydl_opts = _build_ydl_opts(config)
    try:
        await loop.run_in_executor(
            None,
            functools.partial(_sync_download, video.url, ydl_opts),
        )
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
    opts: dict[str, str | int | bool],
) -> dict[str, Any] | None:
    """Extract post metadata without downloading (runs in an executor)."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.extract_info(url, download=False)
        if result is None:
            return None
        return dict(result)


def _build_extract_opts(config: ScraperConfig) -> dict[str, str | int | bool]:
    """Construct yt-dlp options for metadata-only extraction."""
    opts: dict[str, str | int | bool] = {
        "quiet": True,
        "no_warnings": True,
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
    loop: asyncio.AbstractEventLoop,
    *,
    retries: int = 3,
) -> None:
    """Download slideshow images into ``output_dir/<video_id>/``.

    Args:
        image_urls: URLs of the images to download.
        output_dir: Base output directory.
        video_id: The TikTok post ID (used as subdirectory name).
        loop: The running event loop for executor calls.
        retries: Number of retry attempts per image on transient failures.

    Raises:
        Exception: If any image download fails after all retries.
    """
    photo_dir = output_dir / video_id
    photo_dir.mkdir(parents=True, exist_ok=True)

    for idx, url in enumerate(image_urls, start=1):
        ext = _image_extension(url)
        dest = photo_dir / f"image_{idx}{ext}"
        await loop.run_in_executor(
            None,
            functools.partial(_sync_download_file, url, dest, retries=retries),
        )
        logger.debug("Saved %s", dest)


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
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or exc.code >= 500:
                delay = 2**attempt
                logger.warning(
                    "HTTP %d downloading %s (attempt %d/%d), retrying in %ds...",
                    exc.code,
                    url,
                    attempt,
                    retries,
                    delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable HTTP error (4xx other than 429).
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            delay = 2**attempt
            logger.warning(
                "Network error downloading %s (attempt %d/%d): %s, retrying in %ds...",
                url,
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
            continue

    # Exhausted all retries.
    msg = f"Failed to download {url} after {retries} attempts"
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


def _sync_download(url: str, opts: dict[str, str | int | bool]) -> None:
    """Synchronous yt-dlp download (runs inside an executor)."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _build_ydl_opts(config: ScraperConfig) -> dict[str, str | int | bool]:
    """Construct the yt-dlp options dictionary."""
    outtmpl = str(config.output_dir / "%(id)s.%(ext)s")

    opts: dict[str, str | int | bool] = {
        "outtmpl": outtmpl,
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "retries": config.download_retries,
        "fragment_retries": config.download_retries,
        "noprogress": True,
    }

    if config.cookies_file is not None:
        opts["cookiefile"] = str(config.cookies_file)

    return opts
