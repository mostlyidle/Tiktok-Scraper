"""Unit tests for the downloader module."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.downloader import (
    DownloadArtifactError,
    MetadataMismatchError,
    _build_embed_ydl_opts,
    _build_ydl_opts,
    _extract_image_urls,
    _find_completed_artifact,
    _has_video_formats,
    _is_retryable_download_error,
    _photo_variant_url,
    _requires_ffmpeg,
    _SafeImageRedirectHandler,
    _sync_download_embed,
    _sync_download_file,
    _sync_download_info,
    _sync_extract_info,
    _validate_public_image_url,
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

_VALID_JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
_VALID_MP4 = b"\x00\x00\x00\x18ftypisomfake-mp4"
_VALID_MP3 = b"ID3fake-mp3"


def _metadata_side_effect(template: dict[str, object]):
    """Return realistic metadata whose ID matches the requested post URL."""

    def _extract(url: str, _opts: dict[str, object]) -> dict[str, object]:
        post_id = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
        metadata = dict(template)
        metadata["id"] = post_id
        metadata["uploader"] = "testuser"
        metadata["webpage_url"] = url
        return metadata

    return _extract


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


def test_photo_variant_url_preserves_only_the_canonical_post_parts() -> None:
    assert _photo_variant_url(
        "https://www.tiktok.com/@testuser/video/7123456789012345678?secret=value#fragment"
    ) == "https://www.tiktok.com/@testuser/photo/7123456789012345678"


class TestExtractImageUrls:
    """Tests for :func:`_extract_image_urls`."""

    def test_extracts_from_thumbnails(self) -> None:
        urls = _extract_image_urls(  # type: ignore[arg-type]
            _SLIDESHOW_INFO,
            is_photo_post=True,
        )
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
        urls = _extract_image_urls(info, is_photo_post=True)  # type: ignore[arg-type]
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
        urls = _extract_image_urls(info, is_photo_post=True)  # type: ignore[arg-type]
        assert len(urls) == 2

    def test_extracts_from_display_images(self) -> None:
        info: dict[str, object] = {
            "imagePost": {
                "displayImages": [
                    {
                        "urlList": [
                            "https://cdn.example.com/photo1",
                        ],
                    },
                    {
                        "urlList": [
                            "https://cdn.example.com/photo2",
                        ],
                    },
                ]
            }
        }
        assert _extract_image_urls(info, is_photo_post=True) == [
            "https://cdn.example.com/photo1",
            "https://cdn.example.com/photo2",
        ]

    def test_ignores_non_image_thumbnails(self) -> None:
        """Thumbnail URLs that don't look like images should be ignored."""
        info: dict[str, object] = {
            "formats": [{"vcodec": "none"}],
            "thumbnails": [
                {"url": "https://cdn.example.com/thumb.mp4"},
                {"url": "https://cdn.example.com/photo.webp"},
            ],
        }
        urls = _extract_image_urls(info, is_photo_post=True)  # type: ignore[arg-type]
        assert len(urls) == 1
        assert urls[0] == "https://cdn.example.com/photo.webp"

    def test_semantic_image_post_data_wins_over_cover_thumbnails(self) -> None:
        info: dict[str, object] = {
            "imagePost": {
                "images": [
                    {
                        "imageURL": {
                            "urlList": [
                                "https://cdn.example.com/content/no-extension"
                            ]
                        }
                    }
                ]
            },
            "thumbnails": [{"url": "https://cdn.example.com/cover.jpg"}],
        }

        assert _extract_image_urls(info, is_photo_post=True) == [
            "https://cdn.example.com/content/no-extension"
        ]

    def test_audio_post_cover_thumbnails_are_not_slideshow_images(self) -> None:
        info: dict[str, object] = {
            "formats": [{"vcodec": "none", "acodec": "mp3"}],
            "thumbnails": [{"url": "https://cdn.example.com/cover.jpg"}],
        }

        assert _extract_image_urls(info) == []


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
        (scraper_config.output_dir / "7123456789012345678.mp4").write_bytes(_VALID_MP4)
        (scraper_config.output_dir / "7123456789012345679.mp4").write_bytes(_VALID_MP4)

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_dl,
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
        photo_dir = scraper_config.output_dir / "7123456789012345678"
        photo_dir.mkdir()
        (photo_dir / "image_1.jpg").write_bytes(_VALID_JPEG)

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
    async def test_skips_photo_dir_when_discovery_used_video_route(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        photo_dir = scraper_config.output_dir / "7123456789012345678"
        photo_dir.mkdir()
        (photo_dir / "image_1.jpg").write_bytes(_VALID_JPEG)

        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )

        result = await download_videos([video], scraper_config)

        assert result.skipped == 1
        assert result.downloaded == 0

    @pytest.mark.asyncio
    async def test_photo_post_does_not_skip_existing_audio_artifact(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        (scraper_config.output_dir / "7123456789012345678.mp3").write_bytes(_VALID_MP3)
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
        )
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._download_images",
                new=AsyncMock(),
            ),
        ):
            result = await download_videos([video], scraper_config)

        assert result.skipped == 0
        assert result.photos_downloaded == 1

    @pytest.mark.asyncio
    async def test_all_skipped_when_all_exist(
        self,
        scraper_config: ScraperConfig,
        sample_videos: list[VideoInfo],
    ) -> None:
        """When all videos already exist, nothing should be downloaded."""
        scraper_config.output_dir.mkdir(parents=True, exist_ok=True)
        for v in sample_videos:
            (scraper_config.output_dir / f"{v.video_id}.mp4").write_bytes(_VALID_MP4)

        with (
            patch("tiktok_scraper.downloader._sync_extract_info"),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_dl,
        ):
            result = await download_videos(sample_videos, scraper_config)

        assert result.skipped == len(sample_videos)
        assert result.downloaded == 0
        assert mock_dl.call_count == 0

    @pytest.mark.asyncio
    async def test_deduplicates_input_ids_before_download(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        first = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        duplicate = VideoInfo(
            video_id=first.video_id,
            url="https://www.tiktok.com/@other/video/7123456789012345678",
        )
        second = VideoInfo(
            video_id="7123456789012345679",
            url="https://www.tiktok.com/@testuser/video/7123456789012345679",
        )
        config = replace(scraper_config, request_delay=0.0)

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_download,
        ):
            result = await download_videos([first, duplicate, second], config)

        assert result.total_found == 2
        assert result.downloaded == 2
        assert mock_download.call_count == 2

    @pytest.mark.asyncio
    async def test_truthy_non_boolean_artifact_result_fails_closed(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        config = replace(scraper_config, request_delay=0.0)
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=MagicMock(),
            ) as mock_download,
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=RuntimeError("embed permanently failed"),
            ),
        ):
            result = await download_videos([video], config)

        assert result.failed == 1
        mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_metadata_id_fails_closed_without_fallback(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=_VIDEO_INFO,
            ),
            patch("tiktok_scraper.downloader._sync_download_info") as mock_download,
            patch("tiktok_scraper.downloader._sync_download_embed") as mock_embed,
        ):
            result = await download_videos([video], scraper_config)

        assert result.failed == 1
        mock_download.assert_not_called()
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_mismatched_metadata_creator_fails_closed(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        info = {
            "id": video.video_id,
            "uploader": "other",
            "webpage_url": video.url,
            "formats": _VIDEO_INFO["formats"],
        }
        with patch(
            "tiktok_scraper.downloader._sync_extract_info",
            return_value=info,
        ), patch(
            "tiktok_scraper.downloader._sync_download_info",
        ) as mock_download:
            result = await download_videos([video], scraper_config)

        assert result.failed == 1
        assert result.failed_reasons[video.video_id]
        mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_photo_marker_wins_over_video_formats(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        info = {
            "id": video.video_id,
            "uploader": "testuser",
            "webpage_url": video.url,
            "formats": [{"vcodec": "h264", "acodec": "aac"}],
            "imagePost": {"images": [{"imageURL": {"urlList": []}}]},
            "thumbnails": [{"url": "https://cdn.example.com/img.jpg"}],
        }
        with patch(
            "tiktok_scraper.downloader._sync_extract_info",
            return_value=info,
        ), patch(
            "tiktok_scraper.downloader._download_images",
            new=AsyncMock(),
        ) as download_images, patch(
            "tiktok_scraper.downloader._sync_download_info",
        ) as mock_download:
            result = await download_videos([video], scraper_config)

        assert result.photos_downloaded == 1
        download_images.assert_awaited_once()
        mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_symlink_is_rejected_before_work(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "target"
        target.mkdir()
        output = tmp_path / "output-link"
        output.symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError):
            ScraperConfig(
                username="testuser",
                output_dir=output,
                request_delay=0.0,
            )

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
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ),
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
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                side_effect=RuntimeError("network error"),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=RuntimeError("network error"),
            ),
            patch(
                "tiktok_scraper.downloader.asyncio.sleep",
                new=AsyncMock(),
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

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=RuntimeError("extraction error"),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=RuntimeError("network error"),
            ),
            patch(
                "tiktok_scraper.downloader.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            result = await download_videos(videos, scraper_config)

        assert result.failed == 1
        assert result.downloaded == 0

    @pytest.mark.asyncio
    async def test_missing_ffmpeg_fails_only_when_merge_is_required(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        separate_streams = {
            "id": video.video_id,
            "uploader": "testuser",
            "webpage_url": video.url,
            "formats": [
                {"vcodec": "h264", "acodec": "none"},
                {"vcodec": "none", "acodec": "aac"},
            ],
        }
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value=separate_streams,
            ),
            patch("tiktok_scraper.downloader.check_ffmpeg", return_value=False),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_download,
        ):
            result = await download_videos([video], scraper_config)

        assert result.failed == 1
        mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_ffmpeg_uses_progressive_format_when_available(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        )
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                return_value={
                    "id": video.video_id,
                    "uploader": "testuser",
                    "webpage_url": video.url,
                    "formats": [{"vcodec": "h264", "acodec": "aac"}],
                },
            ),
            patch("tiktok_scraper.downloader.check_ffmpeg", return_value=False),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_download,
        ):
            result = await download_videos([video], scraper_config)

        assert result.downloaded == 1
        opts = mock_download.call_args.args[1]
        assert opts["format"] == "best[vcodec!=none][acodec!=none]/best"
        assert "merge_output_format" not in opts
    @pytest.mark.asyncio
    async def test_uses_embed_fallback_after_extract_failure(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A failed post extraction should fall back to TikTok's embed page."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        def save_embed(_video_id: str, opts: dict[str, object]) -> None:
            paths = opts["paths"]
            assert isinstance(paths, dict)
            Path(str(paths["home"]) + "/" + str(opts["outtmpl"])).write_bytes(
                _VALID_MP4
            )

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=RuntimeError("extraction error"),
            ) as mock_extract,
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=save_embed,
            ) as mock_embed,
        ):
            result = await download_videos(videos, scraper_config)

        assert result.downloaded == 1
        assert result.failed == 0
        mock_extract.assert_called_once()
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_exhausted_post_retries_counts_as_failure(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A post should fail after exhausting configured retry attempts."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]
        config = ScraperConfig(
            username=scraper_config.username,
            output_dir=scraper_config.output_dir,
            cookies_file=scraper_config.cookies_file,
            limit=scraper_config.limit,
            headless=scraper_config.headless,
            max_concurrent_downloads=scraper_config.max_concurrent_downloads,
            scroll_delay_min=scraper_config.scroll_delay_min,
            scroll_delay_max=scraper_config.scroll_delay_max,
            download_retries=2,
            video_only=scraper_config.video_only,
            request_delay=0.0,
        )

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=RuntimeError("network still failing"),
            ) as mock_extract,
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=RuntimeError("network error"),
            ),
            patch(
                "tiktok_scraper.downloader.asyncio.sleep",
                new=AsyncMock(),
            ) as mock_sleep,
        ):
            result = await download_videos(videos, config)

        assert result.failed == 1
        assert result.downloaded == 0
        assert mock_extract.call_count == 2
        mock_sleep.assert_awaited_once()

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
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ),
        ):
            result = await download_videos(videos, scraper_config)

        assert result.downloaded == 2
        assert result.failed == 0
        assert result.total_found == 2

    @pytest.mark.asyncio
    async def test_uses_embed_fallback_after_video_download_failure(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A failed primary video download should use the embed source."""
        videos = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]

        def save_embed(_video_id: str, opts: dict[str, object]) -> None:
            paths = opts["paths"]
            assert isinstance(paths, dict)
            Path(str(paths["home"]) + "/" + str(opts["outtmpl"])).write_bytes(
                _VALID_MP4
            )

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                side_effect=RuntimeError("primary download error"),
            ) as mock_download,
            patch(
                "tiktok_scraper.downloader._sync_download_embed",
                side_effect=save_embed,
            ) as mock_embed,
        ):
            result = await download_videos(videos, scraper_config)

        assert result.downloaded == 1
        assert result.failed == 0
        mock_download.assert_called_once()
        mock_embed.assert_called_once()

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

    def test_result_summary_includes_audio_posts(self) -> None:
        result = ScrapeResult(
            username="testuser",
            total_found=4,
            downloaded=1,
            audio_downloaded=1,
            photos_downloaded=1,
            failed=1,
        )
        assert result.success_rate == 75.0

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
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=lambda _url, dest, **_kwargs: dest.write_bytes(_VALID_JPEG),
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
            dest.write_bytes(_VALID_JPEG)

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=fake_download_file,
            ),
        ):
            await download_videos(videos, scraper_config)

        photo_dir = scraper_config.output_dir / "7123456789012345678"
        assert photo_dir.is_dir()
        images = [
            path
            for path in photo_dir.iterdir()
            if path.name != ".tiktok-complete.json"
        ]
        assert len(images) == 2
        assert (photo_dir / ".tiktok-complete.json").is_file()

    @pytest.mark.asyncio
    async def test_slideshow_requires_every_file_before_commit(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
        )
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch("tiktok_scraper.downloader._sync_download_file"),
        ):
            result = await download_videos([video], replace(scraper_config, request_delay=0.0))

        assert result.failed == 1
        assert not (scraper_config.output_dir / video.video_id).exists()
        assert not (
            scraper_config.output_dir / f".{video.video_id}.partial"
        ).exists()

    @pytest.mark.asyncio
    async def test_failed_slideshow_is_not_marked_as_downloaded(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A failed slideshow should retry instead of being skipped on rerun."""
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
        )

        def fail_second_image(url: str, dest: Path, **_kwargs: object) -> None:
            if url.endswith("img2.jpg"):
                raise OSError("network error")
            dest.write_bytes(_VALID_JPEG)

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=fail_second_image,
            ),
        ):
            failed_result = await download_videos([video], scraper_config)

        photo_dir = scraper_config.output_dir / video.video_id
        staging_dir = scraper_config.output_dir / f".{video.video_id}.partial"
        assert failed_result.failed == 1
        assert not photo_dir.exists()
        assert not staging_dir.exists()

        def save_image(_url: str, dest: Path, **_kwargs: object) -> None:
            dest.write_bytes(_VALID_JPEG)

        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=save_image,
            ) as mock_download,
        ):
            retry_result = await download_videos([video], scraper_config)

        assert retry_result.photos_downloaded == 1
        assert mock_download.call_count == 2
        assert photo_dir.is_dir()
        assert not staging_dir.exists()

    @pytest.mark.asyncio
    async def test_cancellation_waits_for_workers_and_removes_staging(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        video = VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/photo/7123456789012345678",
        )
        started = threading.Event()

        def blocking_download(_url: str, dest: Path, **_kwargs: object) -> None:
            started.set()
            time.sleep(0.1)
            dest.write_bytes(_VALID_JPEG)

        config = replace(
            scraper_config,
            max_concurrent_downloads=1,
            request_delay=0.0,
            download_retries=1,
        )
        with (
            patch(
                "tiktok_scraper.downloader._sync_extract_info",
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=blocking_download,
            ),
        ):
            task = asyncio.create_task(download_videos([video], config))
            assert await asyncio.to_thread(started.wait, 2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert not (
            config.output_dir / f".{video.video_id}.partial"
        ).exists()


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
                side_effect=_metadata_side_effect(_AUDIO_ONLY_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_dl,
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
                side_effect=_metadata_side_effect(_AUDIO_ONLY_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ) as mock_dl,
        ):
            result = await download_videos(videos, scraper_config)

        assert result.audio_downloaded == 1
        assert result.downloaded == 0
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
                side_effect=_metadata_side_effect(_SLIDESHOW_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_file",
                side_effect=lambda _url, dest, **_kwargs: dest.write_bytes(_VALID_JPEG),
            ),
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
                side_effect=_metadata_side_effect(_VIDEO_INFO),
            ),
            patch(
                "tiktok_scraper.downloader._sync_download_info",
                return_value=True,
            ),
        ):
            result = await download_videos(videos, video_only_config)

        assert result.downloaded == 1
        assert result.skipped == 0


class TestSyncDownloadFileRetry:
    """Tests for :func:`_sync_download_file` retry logic."""

    def test_rejects_private_dns_resolution(self) -> None:
        with (
            patch(
                "tiktok_scraper.downloader.socket.getaddrinfo",
                return_value=[
                    ("family", "socktype", "proto", "canon", ("127.0.0.1", 443))
                ],
            ),
            pytest.raises(ValueError, match="private"),
        ):
            _validate_public_image_url("https://cdn.example.com/img.jpg")

    def test_succeeds_on_first_attempt(self, tmp_path: Path) -> None:
        """File should be written when the download succeeds immediately."""
        dest = tmp_path / "image.jpg"

        with patch("tiktok_scraper.downloader._open_image_url") as mock_open:
            mock_resp = mock_open.return_value.__enter__.return_value
            mock_resp.read.side_effect = [_VALID_JPEG, b""]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == _VALID_JPEG
        assert not dest.with_name("image.jpg.part").exists()

    def test_retries_on_http_503(self, tmp_path: Path) -> None:
        """Should retry on HTTP 503 and succeed on a subsequent attempt."""
        dest = tmp_path / "image.jpg"

        error_503 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 503, "Service Unavailable", {}, None
        )

        with (
            patch("tiktok_scraper.downloader._open_image_url") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            # First call raises 503, second succeeds.
            success_resp = io.BytesIO(_VALID_JPEG)
            ctx_mgr.side_effect = [error_503, success_resp]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == _VALID_JPEG

    def test_retries_on_http_429(self, tmp_path: Path) -> None:
        """Should retry on HTTP 429 (rate limit)."""
        dest = tmp_path / "image.jpg"

        error_429 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 429, "Too Many Requests", {}, None
        )

        with (
            patch("tiktok_scraper.downloader._open_image_url") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            success_resp = io.BytesIO(_VALID_JPEG)
            ctx_mgr.side_effect = [error_429, success_resp]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == _VALID_JPEG

    def test_no_retry_on_http_404(self, tmp_path: Path) -> None:
        """HTTP 404 is not retryable and should raise immediately."""
        dest = tmp_path / "image.jpg"

        error_404 = urllib.error.HTTPError(
            "https://cdn.example.com/img.jpg", 404, "Not Found", {}, None
        )

        with (
            patch("tiktok_scraper.downloader._open_image_url") as mock_open,
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
            patch("tiktok_scraper.downloader._open_image_url") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            mock_open.return_value.__enter__.side_effect = error_503
            with pytest.raises(urllib.error.URLError, match="after 2 attempts"):
                _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=2)

    def test_retries_on_network_error(self, tmp_path: Path) -> None:
        """Should retry on transient network errors (URLError, OSError)."""
        dest = tmp_path / "image.jpg"

        with (
            patch("tiktok_scraper.downloader._open_image_url") as mock_open,
            patch("tiktok_scraper.downloader.time.sleep"),
        ):
            ctx_mgr = mock_open.return_value.__enter__
            success_resp = io.BytesIO(_VALID_JPEG)
            ctx_mgr.side_effect = [
                urllib.error.URLError("Connection reset"),
                success_resp,
            ]
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        assert dest.read_bytes() == _VALID_JPEG

    def test_rejects_oversized_image_before_writing(self, tmp_path: Path) -> None:
        dest = tmp_path / "image.jpg"
        response = MagicMock()
        response.headers = {"Content-Length": "101"}

        with patch(
            "tiktok_scraper.downloader._open_image_url",
            return_value=response,
        ):
            response.__enter__.return_value = response
            with pytest.raises(DownloadArtifactError, match="limit"):
                _sync_download_file(
                    "https://cdn.example.com/img.jpg",
                    dest,
                    max_bytes=100,
                )

        assert not dest.exists()

    def test_sends_only_scoped_cookies_to_image_request(self, tmp_path: Path) -> None:
        dest = tmp_path / "image.jpg"
        cookies = tmp_path / "cookies.txt"
        cookies.write_text(
            ".tiktok.com\tTRUE\t/\tTRUE\t0\tsid\tsecret\n"
            ".example.com\tTRUE\t/\tTRUE\t0\tother\tprivate\n",
            encoding="utf-8",
        )
        response = io.BytesIO(_VALID_JPEG)

        with patch(
            "tiktok_scraper.downloader._open_image_url",
            return_value=response,
        ) as mock_open:
            _sync_download_file(
                "https://cdn.tiktok.com/img.jpg",
                dest,
                cookies_file=cookies,
            )

        request = mock_open.call_args.args[0]
        assert request.get_header("Cookie") == "sid=secret"

    def test_redirect_handler_does_not_forward_cookies_cross_host(self) -> None:
        from urllib.request import Request

        handler = _SafeImageRedirectHandler(None)
        request = Request(
            "https://cdn.tiktok.com/img.jpg",
            headers={"Cookie": "sid=secret"},
        )
        with patch(
            "tiktok_scraper.downloader._validate_public_image_url"
        ) as validate_url:
            redirected = handler.redirect_request(
                request,
                MagicMock(),
                302,
                "Found",
                {"Location": "https://cdn.example.com/img.jpg"},
                "https://cdn.example.com/img.jpg",
            )

        assert redirected is not None
        assert redirected.get_header("Cookie") is None
        validate_url.assert_called_once_with("https://cdn.example.com/img.jpg")

    def test_local_filesystem_errors_are_not_retried(self, tmp_path: Path) -> None:
        dest = tmp_path / "image.jpg"
        assert not _is_retryable_download_error(PermissionError("permission denied"))
        assert _is_retryable_download_error(urllib.error.URLError("network"))
        with (
            patch(
                "tiktok_scraper.downloader._open_image_url",
                side_effect=PermissionError("permission denied"),
            ),
            patch("tiktok_scraper.downloader.time.sleep") as sleep,
            pytest.raises(PermissionError),
        ):
            _sync_download_file("https://cdn.example.com/img.jpg", dest, retries=3)

        sleep.assert_not_called()


class TestPreExtractedDownload:
    """Tests for downloading yt-dlp metadata without another URL extraction."""

    def test_yt_dlp_receives_an_isolated_cookie_file(self, tmp_path: Path) -> None:
        cookies = tmp_path / "cookies.txt"
        original = b".tiktok.com\tTRUE\t/\tFALSE\t0\tid\tvalue\n"
        cookies.write_bytes(original)
        seen_cookie_paths: list[Path] = []

        def make_ydl(opts: dict[str, object]) -> MagicMock:
            isolated = Path(str(opts["cookiefile"]))
            seen_cookie_paths.append(isolated)
            assert isolated != cookies
            assert isolated.read_bytes() == original
            ydl = MagicMock()
            ydl.__enter__.return_value.extract_info.return_value = {"ok": True}
            return ydl

        opts = {"quiet": True, "cookiefile": str(cookies)}
        with patch(
            "tiktok_scraper.downloader.yt_dlp.YoutubeDL",
            side_effect=make_ydl,
        ):
            assert _sync_extract_info("https://www.tiktok.com/", opts) == {"ok": True}

        assert len(seen_cookie_paths) == 1
        assert cookies.read_bytes() == original

    def test_processes_preextracted_info(self) -> None:
        opts = {"quiet": True}
        with patch("tiktok_scraper.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            ydl = mock_ydl.return_value.__enter__.return_value
            _sync_download_info(_VIDEO_INFO, opts)  # type: ignore[arg-type]

        ydl.process_ie_result.assert_called_once_with(_VIDEO_INFO, download=True)
        ydl.extract_info.assert_not_called()

    def test_photo_metadata_uses_embed_hydration_instead_of_ytdlp(self) -> None:
        video_id = "7123456789012345678"
        photo_url = f"https://www.tiktok.com/@testuser/photo/{video_id}"
        state = {
            "source": {
                "data": {
                    f"/embed/v2/{video_id}": {
                        "videoData": {
                            "itemInfos": {"id": video_id},
                            "authorInfos": {"uniqueId": "testuser"},
                            "imagePostInfo": {
                                "images": [
                                    {
                                        "imageURL": {
                                            "urlList": [
                                                "https://cdn.example.com/img.jpg"
                                            ]
                                        }
                                    }
                                ]
                            },
                        }
                    }
                }
            }
        }
        response = MagicMock()
        response.url = f"https://www.tiktok.com/embed/v2/{video_id}"
        response.text = (
            '<script id="__FRONTITY_CONNECT_STATE__">'
            f"{json.dumps(state)}"
            "</script>"
        )
        opts = {"quiet": True, "socket_timeout": 30.0}
        with patch(
            "tiktok_scraper.downloader.curl_requests.get",
            return_value=response,
        ) as get:
            info = _sync_extract_info(photo_url, opts)

        assert info is not None
        assert info["id"] == video_id
        assert info["uploader"] == "testuser"
        assert info["imagePost"]
        get.assert_called_once()

    def test_video_route_promotes_photo_metadata_when_ytdlp_cannot_extract(self) -> None:
        video_id = "7123456789012345678"
        video_url = f"https://www.tiktok.com/@testuser/video/{video_id}"
        state = {
            "source": {
                "data": {
                    f"/embed/v2/{video_id}": {
                        "videoData": {
                            "itemInfos": {"id": video_id},
                            "authorInfos": {"uniqueId": "testuser"},
                            "imagePostInfo": {
                                "displayImages": [
                                    {
                                        "urlList": [
                                            "https://cdn.example.com/photo"
                                        ]
                                    }
                                ]
                            },
                        }
                    }
                }
            }
        }
        response = MagicMock()
        response.status_code = 200
        response.url = f"https://www.tiktok.com/embed/v2/{video_id}"
        response.text = (
            '<script id="__FRONTITY_CONNECT_STATE__">'
            f"{json.dumps(state)}"
            "</script>"
        )
        opts = {"quiet": True, "socket_timeout": 30.0}
        with (
            patch("tiktok_scraper.downloader.yt_dlp.YoutubeDL") as mock_ydl,
            patch(
                "tiktok_scraper.downloader.curl_requests.get",
                return_value=response,
            ),
        ):
            mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
                RuntimeError("photo route unsupported")
            )
            info = _sync_extract_info(video_url, opts)

        assert info is not None
        assert info["id"] == video_id
        assert info["imagePost"]

    def test_preextracted_download_commits_from_private_staging(
        self,
        tmp_path: Path,
    ) -> None:
        video_id = "7123456789012345678"
        info = {"id": video_id, "formats": _VIDEO_INFO["formats"]}
        opts = {
            "quiet": True,
            "outtmpl": f"{video_id}.%(ext)s",
            "paths": {"home": str(tmp_path)},
        }

        def make_ydl(received_opts: dict[str, object]) -> MagicMock:
            ydl = MagicMock()

            def process(_info: dict[str, object], *, download: bool) -> None:
                assert download
                paths = received_opts["paths"]
                assert isinstance(paths, dict)
                staged = Path(str(paths["home"])) / f"{video_id}.mp4"
                staged.write_bytes(_VALID_MP4)

            ydl.__enter__.return_value.process_ie_result.side_effect = process
            return ydl

        with patch(
            "tiktok_scraper.downloader.yt_dlp.YoutubeDL",
            side_effect=make_ydl,
        ):
            assert _sync_download_info(info, opts, video_id, "video")

        assert (tmp_path / f"{video_id}.mp4").read_bytes() == _VALID_MP4
        assert not (tmp_path / f".{video_id}.download").exists()


class TestEmbedFallback:
    """Tests for the TikTok embed download fallback."""

    def test_downloads_first_embed_source(self) -> None:
        opts = {"quiet": True}
        with patch("tiktok_scraper.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            ydl = mock_ydl.return_value.__enter__.return_value
            _sync_download_embed("7123456789012345678", opts)

        ydl.extract_info.assert_called_once_with(
            "https://www.tiktok.com/embed/v2/7123456789012345678",
            download=True,
        )

    def test_embed_rejects_wrong_metadata_before_publishing(
        self,
        tmp_path: Path,
    ) -> None:
        video_id = "7123456789012345678"
        config = ScraperConfig(username="testuser", output_dir=tmp_path)
        opts = _build_embed_ydl_opts(config, video_id)
        opts["_expected_author"] = "testuser"
        with patch(
            "tiktok_scraper.downloader.yt_dlp.YoutubeDL"
        ) as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = {
                "id": video_id,
                "uploader": "other",
                "webpage_url": f"https://www.tiktok.com/@other/video/{video_id}",
            }
            with pytest.raises(MetadataMismatchError):
                _sync_download_embed(video_id, opts)

        assert not (tmp_path / f"{video_id}.mp4").exists()
        assert not (tmp_path / f".{video_id}.download").exists()

    def test_embed_commits_only_an_id_matching_result(self, tmp_path: Path) -> None:
        video_id = "7123456789012345678"
        config = ScraperConfig(username="testuser", output_dir=tmp_path)
        opts = _build_embed_ydl_opts(config, video_id)
        opts["_expected_author"] = "testuser"

        def make_ydl(received_opts: dict[str, object]) -> MagicMock:
            ydl = MagicMock()

            def extract(_url: str, *, download: bool) -> dict[str, object]:
                assert download
                paths = received_opts["paths"]
                assert isinstance(paths, dict)
                Path(str(paths["home"]) + f"/{video_id}.mp4").write_bytes(_VALID_MP4)
                return {
                    "id": video_id,
                    "webpage_url": f"https://www.tiktok.com/embed/v2/{video_id}",
                }

            ydl.__enter__.return_value.extract_info.side_effect = extract
            return ydl

        with patch(
            "tiktok_scraper.downloader.yt_dlp.YoutubeDL",
            side_effect=make_ydl,
        ):
            result = _sync_download_embed(video_id, opts)

        assert result is not None
        assert (tmp_path / f"{video_id}.mp4").read_bytes() == _VALID_MP4


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

    def test_ytdlp_options_construct_with_current_version(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Options must be accepted by the installed yt-dlp API."""
        import yt_dlp

        opts = _build_ydl_opts(scraper_config)
        assert str(opts["impersonate"]) == "chrome"
        with yt_dlp.YoutubeDL(opts):
            pass

    def test_progressive_options_do_not_request_a_merge(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        opts = _build_ydl_opts(scraper_config, merge=False)
        assert "merge_output_format" not in opts
        assert opts["format"] == "best[vcodec!=none][acodec!=none]/best"

    def test_extractor_retries_match_config(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Extractor retries should follow the configured retry count."""
        opts = _build_ydl_opts(scraper_config)
        assert opts["extractor_retries"] == 0

    def test_embed_fallback_uses_one_fixed_mp4_output(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """The embed page's duplicate source should be downloaded only once."""
        video_id = "7123456789012345678"
        opts = _build_embed_ydl_opts(scraper_config, video_id)

        assert opts["outtmpl"] == f"{video_id}.mp4"
        assert opts["paths"] == {"home": str(scraper_config.output_dir.resolve())}
        assert opts["playlistend"] == 1
        assert opts["continuedl"] is False
        assert opts["overwrites"] is True
        assert str(opts["impersonate"]) == "chrome"


class TestArtifactSemantics:
    """Completed artifacts must be nonempty and never temporary."""

    def test_empty_and_temporary_files_are_not_artifacts(self, tmp_path: Path) -> None:
        assert _find_completed_artifact(tmp_path, "7123456789012345678") is None
        (tmp_path / "7123456789012345678.mp4.part").write_bytes(b"partial")
        (tmp_path / "7123456789012345678.mp4").touch()
        assert _find_completed_artifact(tmp_path, "7123456789012345678") is None

    def test_artifacts_are_media_aware(self, tmp_path: Path) -> None:
        video_id = "7123456789012345678"
        (tmp_path / f"{video_id}.mp4").write_bytes(_VALID_MP4)
        (tmp_path / f"{video_id}.mp3").write_bytes(_VALID_MP3)

        assert _find_completed_artifact(tmp_path, video_id, media_type="video") == (
            tmp_path / f"{video_id}.mp4"
        )
        assert _find_completed_artifact(tmp_path, video_id, media_type="audio") == (
            tmp_path / f"{video_id}.mp3"
        )
        assert _find_completed_artifact(tmp_path, video_id, media_type="image") is None

    def test_requires_ffmpeg_for_separate_streams(self) -> None:
        assert _requires_ffmpeg(
            {
                "formats": [
                    {"vcodec": "h264", "acodec": "none"},
                    {"vcodec": "none", "acodec": "aac"},
                ]
            }
        )
        assert not _requires_ffmpeg({"formats": [{"vcodec": "h264", "acodec": "aac"}]})


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
