"""Unit tests for the downloader module."""

from __future__ import annotations

import logging
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.downloader import (
    _build_ydl_opts,
    _extract_image_urls,
    _has_video_formats,
    _sync_download_file,
    download_videos,
)
from tiktok_scraper.models import ScrapeResult, VideoInfo

# ---------------------------------------------------------------------------
# Helpers — fake yt-dlp metadata dicts
# ---------------------------------------------------------------------------

_VIDEO_INFO: dict[str, object] = {
    "formats": [
        {"format_id": "0", "vcodec": "h264", "acodec": "aac"},
    ],
}

_AUDIO_ONLY_INFO: dict[str, object] = {
    "formats": [
        {"format_id": "0", "vcodec": "none", "acodec": "mp3"},
    ],
}

_SLIDESHOW_INFO: dict[str, object] = {
    "formats": [
        {"format_id": "0", "vcodec": "none", "acodec": "mp3"},
    ],
    "thumbnails": [
        {"url": "https://p16.tiktokcdn.com/img1.jpg"},
        {"url": "https://p16.tiktokcdn.com/img2.jpg"},
    ],
}


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHasVideoFormats:
    """Tests for :func:`_has_video_formats`."""

    def test_video_post(self) -> None:
        assert _has_video_formats(_VIDEO_INFO) is True  # type: ignore[arg-type]

    def test_audio_only_post(self) -> None:
        assert _has_video_formats(_AUDIO_ONLY_INFO) is False  # type: ignore[arg-type]

    def test_slideshow_post(self) -> None:
        assert _has_video_formats(_SLIDESHOW_INFO) is False  # type: ignore[arg-type]

    def test_no_formats_key_with_vcodec(self) -> None:
        info: dict[str, object] = {"vcodec": "h264"}
        assert _has_video_formats(info) is True  # type: ignore[arg-type]

    def test_no_formats_key_without_vcodec(self) -> None:
        info: dict[str, object] = {"vcodec": "none"}
        assert _has_video_formats(info) is False  # type: ignore[arg-type]

    def test_empty_formats_list(self) -> None:
        info: dict[str, object] = {"formats": []}
        assert _has_video_formats(info) is False  # type: ignore[arg-type]


class TestExtractImageUrls:
    """Tests for :func:`_extract_image_urls`."""

    def test_extracts_from_thumbnails(self) -> None:
        urls = _extract_image_urls(_SLIDESHOW_INFO)  # type: ignore[arg-type]
        assert len(urls) == 2
        assert "https://p16.tiktokcdn.com/img1.jpg" in urls

    def test_no_images_for_video(self) -> None:
        urls = _extract_image_urls(_VIDEO_INFO)  # type: ignore[arg-type]
        assert urls == []

    def test_no_images_for_audio_only(self) -> None:
        urls = _extract_image_urls(_AUDIO_ONLY_INFO)  # type: ignore[arg-type]
        assert urls == []

    def test_deduplicates_urls(self) -> None:
        info: dict[str, object] = {
            "formats": [{"vcodec": "none", "acodec": "mp3"}],
            "thumbnails": [
                {"url": "https://cdn.example.com/img.jpg"},
                {"url": "https://cdn.example.com/img.jpg"},
            ],
        }
        urls = _extract_image_urls(info)  # type: ignore[arg-type]
        assert len(urls) == 1

    def test_extracts_from_image_post(self) -> None:
        info: dict[str, object] = {
            "formats": [{"vcodec": "none"}],
            "imagePost": {
                "images": [
                    {
                        "imageURL": {
                            "urlList": [
                                "https://cdn.example.com/photo1.jpeg",
                            ],
                        },
                    },
                    {
                        "imageURL": {
                            "urlList": [
                                "https://cdn.example.com/photo2.jpeg",
                            ],
                        },
                    },
                ],
            },
        }
        urls = _extract_image_urls(info)  # type: ignore[arg-type]
        assert len(urls) == 2

    def test_ignores_non_image_thumbnails(self) -> None:
        """Thumbnail URLs that don't look like images should be ignored."""
        info: dict[str, object] = {
            "formats": [{"vcodec": "none"}],
            "thumbnails": [
                {"url": "https://cdn.example.com/thumb.mp4"},
                {"url": "https://cdn.example.com/photo.webp"},
            ],
        }
        urls = _extract_image_urls(info)  # type: ignore[arg-type]
        assert len(urls) == 1
        assert urls[0] == "https://cdn.example.com/photo.webp"


# ---------------------------------------------------------------------------
# Integration-style tests for download_videos
# ---------------------------------------------------------------------------


class TestDownloadVideos:
    """Tests for :func:`download_videos`."""

    @pytest.mark.asyncio
    async def test_skips_already_downloaded(
        self,
        scraper_config: ScraperConfig,
        sample_videos: list[VideoInfo],
    ) -> None:
        """Videos whose ID matches an existing file should be skipped."""
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        (scraper_config.output_dir / "7123456789012345678.mp4").touch()
        (scraper_config.output_dir / "7123456789012345679.mp4").touch()

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download") as mock_dl,
        ):
            result = await download_videos(sample_videos, scraper_config)

        assert result.skipped == 2
        assert mock_dl.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_already_downloaded_photo_dirs(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Photo slideshow directories should count as already downloaded."""
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        (scraper_config.output_dir / "7123456789012345678").mkdir()

        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
            ),
        ]

        result = await download_videos(videos, scraper_config)

        assert result.skipped == 1
        assert result.downloaded == 0

    @pytest.mark.asyncio
    async def test_all_skipped_when_all_exist(
        self,
        scraper_config: ScraperConfig,
        sample_videos: list[VideoInfo],
    ) -> None:
        """When all videos already exist, nothing should be downloaded."""
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        for v in sample_videos:
            (scraper_config.output_dir / f"{v.video_id}.mp4").touch()

        with (
            patch("tiktok_scraper.downloader._sync_extract_info"),
            patch("tiktok_scraper.downloader._sync_download") as mock_dl,
        ):
            result = await download_videos(sample_videos, scraper_config)

        assert result.skipped == len(sample_videos)
        assert result.downloaded == 0
        assert mock_dl.call_count == 0

    @pytest.mark.asyncio
    async def test_creates_output_dir(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Output directory should be created if it doesn't exist."""
        assert not scraper_config.output_dir.exists()

        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download"),
        ):
            await download_videos(videos, scraper_config)

        assert scraper_config.output_dir.exists()

    @pytest.mark.asyncio
    async def test_failed_download_counted(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A download that raises should be counted as failed."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch(
                "tiktok_scraper.downloader._sync_download",
                side_effect=RuntimeError("network error"),
            ),
        ):
            result = await download_videos(videos, scraper_config)

        assert result.failed == 1
        assert result.downloaded == 0
        assert "7123456789012345678" in result.failed_videos

    @pytest.mark.asyncio
    async def test_failed_extract_info_counted(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A metadata extraction failure should be counted as failed."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with patch(
            "tiktok_scraper.downloader._sync_extract_info",
            side_effect=RuntimeError("extraction error"),
        ):
            result = await download_videos(videos, scraper_config)

        assert result.failed == 1
        assert result.downloaded == 0

    @pytest.mark.asyncio
    async def test_successful_download_counted(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Successful downloads should be reflected in the result."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
            VideoInfo(
                video_id="7123456789012345679",
                url="https://www.tiktok.com/@testuser/video/7123456789012345679",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download"),
        ):
            result = await download_videos(videos, scraper_config)

        assert result.downloaded == 2
        assert result.failed == 0
        assert result.total_found == 2

    @pytest.mark.asyncio
    async def test_result_summary(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """ScrapeResult should correctly calculate success_rate."""
        result = ScrapeResult(
            username="testuser",
            total_found=10,
            downloaded=8,
            skipped=0,
            failed=2,
        )
        assert result.success_rate == 80.0

    @pytest.mark.asyncio
    async def test_result_summary_with_photos(self) -> None:
        """success_rate should include photos_downloaded."""
        result = ScrapeResult(
            username="testuser",
            total_found=10,
            downloaded=6,
            photos_downloaded=2,
            skipped=0,
            failed=2,
        )
        assert result.success_rate == 80.0

    @pytest.mark.asyncio
    async def test_empty_video_list(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """An empty video list should produce a zero-count result."""
        result = await download_videos([], scraper_config)
        assert result.total_found == 0
        assert result.downloaded == 0
        assert result.skipped == 0


class TestPhotoSlideshow:
    """Tests for photo slideshow downloading."""

    @pytest.mark.asyncio
    async def test_slideshow_downloads_images(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Posts with images and no video should download photos."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_SLIDESHOW_INFO,
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
            ) as mock_dl_file,
        ):
            result = await download_videos(videos, scraper_config)

        assert result.photos_downloaded == 1
        assert result.downloaded == 0
        # Two images in _SLIDESHOW_INFO.
        assert mock_dl_file.call_count == 2

    @pytest.mark.asyncio
    async def test_slideshow_creates_subdirectory(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Slideshow images should be saved in a <video_id>/ subdirectory."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
            ),
        ]

        def fake_download_file(url: str, dest: Path, **_kwargs: object) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake image data")

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_SLIDESHOW_INFO,
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=fake_download_file,
            ),
        ):
            await download_videos(videos, scraper_config)

        photo_dir = scraper_config.output_dir / "7123456789012345678"
        assert photo_dir.is_dir()
        images = list(photo_dir.iterdir())
        assert len(images) == 2


class TestVideoOnly:
    """Tests for the --video-only flag."""

    @pytest.mark.asyncio
    async def test_audio_only_skipped(
        self,
        video_only_config: ScraperConfig,
    ) -> None:
        """Audio-only posts with no images should be skipped."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_AUDIO_ONLY_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download") as mock_dl,
        ):
            result = await download_videos(videos, video_only_config)

        assert result.skipped_audio_only == 1
        assert result.skipped == 0
        assert result.downloaded == 0
        assert mock_dl.call_count == 0

    @pytest.mark.asyncio
    async def test_audio_only_downloaded_when_not_video_only(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Audio-only posts should be downloaded when video_only is False."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_AUDIO_ONLY_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download") as mock_dl,
        ):
            result = await download_videos(videos, scraper_config)

        assert result.downloaded == 1
        assert mock_dl.call_count == 1

    @pytest.mark.asyncio
    async def test_slideshows_still_downloaded_with_video_only(
        self,
        video_only_config: ScraperConfig,
    ) -> None:
        """Slideshows with images should still be downloaded in video-only mode."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_SLIDESHOW_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download_file"),
        ):
            result = await download_videos(videos, video_only_config)

        assert result.photos_downloaded == 1
        assert result.skipped == 0

    @pytest.mark.asyncio
    async def test_videos_still_downloaded_with_video_only(
        self,
        video_only_config: ScraperConfig,
    ) -> None:
        """Regular video posts should still be downloaded in video-only mode."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download"),
        ):
            result = await download_videos(videos, video_only_config)

        assert result.downloaded == 1
        assert result.skipped == 0


class TestSyncDownloadFileRetry:
    """Tests for :func:`_sync_download_file` retry logic."""

    def test_succeeds_on_first_attempt(self, tmp_path: Path) -> None:
        """File should be written when the download succeeds immediately."""
        dest = tmp_path / "image.jpg"

        with patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open:
            mock_resp = mock_open.return_value.__enter__.return_value
            mock_resp.read.return_value = b"fake image"
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == b"fake image"

    def test_retries_on_http_503(self, tmp_path: Path) -> None:
        """Should retry on HTTP 503 and succeed on a subsequent attempt."""
        dest = tmp_path / "image.jpg"

        error_503 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 503, "Service Unavailable", {}, None
        )

        with (
            patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            # First call raises 503, second succeeds.
            success_resp = type("Resp", (), {"read": lambda self: b"ok"})()
            ctx_mgr.side_effect = [error_503, success_resp]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == b"ok"

    def test_retries_on_http_429(self, tmp_path: Path) -> None:
        """Should retry on HTTP 429 (rate limit)."""
        dest = tmp_path / "image.jpg"

        error_429 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 429, "Too Many Requests", {}, None
        )

        with (
            patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            success_resp = type("Resp", (), {"read": lambda self: b"ok"})()
            ctx_mgr.side_effect = [error_429, success_resp]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == b"ok"

    def test_no_retry_on_http_404(self, tmp_path: Path) -> None:
        """HTTP 404 is not retryable and should raise immediately."""
        dest = tmp_path / "image.jpg"

        error_404 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 404, "Not Found", {}, None
        )

        with (
            patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep") as mock_sleep,
        ):
            mock_open.return_value.__enter__.side_effect = error_404
            with pytest.raises(urllib.error.HTTPError, match="404"):
                _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        # Should not have slept (no retry).
        mock_sleep.assert_not_called()

    def test_exhausts_retries(self, tmp_path: Path) -> None:
        """Should raise URLError after exhausting all retry attempts."""
        dest = tmp_path / "image.jpg"

        error_503 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 503, "Service Unavailable", {}, None
        )

        with (
            patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            mock_open.return_value.__enter__.side_effect = error_503
            with pytest.raises(urllib.error.URLError, match="after 2 attempts"):
                _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=2)

    def test_retries_on_network_error(self, tmp_path: Path) -> None:
        """Should retry on transient network errors (URLError, OSError)."""
        dest = tmp_path / "image.jpg"

        with (
            patch("tiktok_scraper.downloader.urllib.request.urlopen") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            success_resp = type("Resp", (), {"read": lambda self: b"ok"})()
            ctx_mgr.side_effect = [
                urllib.error.URLError("Connection reset"),
                success_resp,
            ]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == b"ok"


class TestBuildYdlOpts:
    """Tests for :func:`_build_ydl_opts` yt-dlp option construction."""

    def test_format_merges_video_and_audio(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Format string should request best video+audio with merge fallback."""
        opts = _build_ydl_opts(scraper_config)
        assert opts["format"] == "bestvideo*+bestaudio/best"

    def test_merge_output_format_is_mp4(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Merged output should always be MP4."""
        opts = _build_ydl_opts(scraper_config)
        assert opts["merge_output_format"] == "mp4"

    def test_cookies_file_included_when_set(self, tmp_path: Path) -> None:
        """cookiefile should be present when cookies_file is configured."""
        cookies = tmp_path / "cookies.txt"
        cookies.touch()
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            cookies_file=cookies,
        )
        opts = _build_ydl_opts(config)
        assert opts["cookiefile"] == str(cookies)

    def test_cookies_file_absent_by_default(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """cookiefile should not be present when cookies_file is None."""
        opts = _build_ydl_opts(scraper_config)
        assert "cookiefile" not in opts


class TestCheckFfmpeg:
    """Tests for :func:`check_ffmpeg` runtime dependency check."""

    def test_returns_true_when_ffmpeg_found(self) -> None:
        """Should return True when ffmpeg is on PATH."""
        with patch(
            "tiktok_scraper.utils.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ):
            from tiktok_scraper.utils import check_ffmpeg

            assert check_ffmpeg() is True

    def test_returns_false_when_ffmpeg_missing(self) -> None:
        """Should return False when ffmpeg is not on PATH."""
        with patch("tiktok_scraper.utils.shutil.which", return_value=None):
            from tiktok_scraper.utils import check_ffmpeg

            assert check_ffmpeg() is False

    def test_logs_warning_when_ffmpeg_missing(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log a warning when ffmpeg is not found."""
        with patch("tiktok_scraper.utils.shutil.which", return_value=None):
            from tiktok_scraper.utils import check_ffmpeg

            with caplog.at_level(logging.WARNING, logger="tiktok_scraper.utils"):
                check_ffmpeg()

        assert "ffmpeg not found" in caplog.text
