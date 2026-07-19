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
                functools.partial(_sync_download_info, info, ydl_opts),
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
    ydl_opts = _build_ydl_opts(config)
    try:
        await loop.run_in_executor(
            None,
            functools.partial(_sync_download_info, info, ydl_opts),
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
        *(
            _download_one(idx, url)
            for idx, url in enumerate(image_urls, start=1)
        ),
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
    for attempt in range(1, retries + 1):
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


def _sync_download_info(
    info: dict[str, Any],
    opts: dict[str, str | int | bool],
) -> None:
    """Download pre-extracted yt-dlp metadata without resolving its URL again."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.process_ie_result(info, download=True)


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
