"""Unit tests for the scraper module."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.exceptions import BrowserError, CreatorNotFoundError
from tiktok_scraper.models import VideoInfo
from tiktok_scraper.scraper import (
    _attach_api_interceptor,
    _collect_from_dom,
    _extract_embed_channel_id,
    _extract_embed_profile_post_ids,
    _extract_hydration_data,
    _fetch_via_browser,
    _fetch_via_embed_channel,
    _fetch_via_ytdlp,
    _html_indicates_inconclusive,
    _is_creator_not_found,
    _is_retryable_ytdlp,
    _parse_hydration_items,
    _redact_log_value,
    _sync_extract,
    _title_indicates_inconclusive,
    _video_infos_from_info,
    fetch_video_urls,
)


def _frontity_html(route: str, data: dict[str, object]) -> str:
    """Build a minimal Frontity state script for parser tests."""
    state = {"source": {"data": {route: data}}}
    return (
        '<script id="__FRONTITY_CONNECT_STATE__" type="application/json">'
        f"{json.dumps(state)}"
        "</script>"
    )


class TestFetchViaYtdlp:
    """Tests for the yt-dlp discovery strategy."""

    @pytest.mark.asyncio
    async def test_discovers_videos(self, scraper_config: ScraperConfig) -> None:
        """Should extract VideoInfo objects from yt-dlp playlist entries."""
        fake_info = {
            "_type": "playlist",
            "entries": [
                {
                    "id": "7123456789012345678",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345678",
                },
                {
                    "id": "7123456789012345679",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345679",
                },
            ],
        }
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value=fake_info,
        ):
            result = await _fetch_via_ytdlp(scraper_config)

        assert len(result) == 2
        ids = {v.video_id for v in result}
        assert "7123456789012345678" in ids
        assert "7123456789012345679" in ids

    @pytest.mark.asyncio
    async def test_validates_authors_urls_ids_and_preserves_source_order(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Only requested-creator posts with canonical IDs should survive."""
        first_id = "7123456789012345685"
        second_id = "7123456789012345681"
        fake_info = {
            "entries": [
                {
                    "id": first_id,
                    "url": (
                        "https://tiktok.com/@testuser/video/"
                        f"{first_id}?token=do-not-keep"
                    ),
                },
                {
                    "id": second_id,
                    "url": f"https://www.tiktok.com/@testuser/video/{second_id}",
                },
                {
                    "id": "7123456789012345682",
                    "url": "https://www.tiktok.com/@other/video/7123456789012345682",
                },
                {
                    "id": "7123456789012345686",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345686",
                    "uploader": "other",
                },
                {
                    "id": "7123456789012345683",
                    "url": "https://example.com/@testuser/video/7123456789012345683",
                },
                {
                    "id": "1234",
                    "url": "https://www.tiktok.com/@testuser/video/1234",
                },
                {
                    "id": "7123456789012345684",
                    "url": "https://www.tiktok.com/@testuser/video/not-numeric",
                },
            ]
        }

        result = _video_infos_from_info(fake_info, None, scraper_config.username)

        assert [video.video_id for video in result] == [first_id, second_id]
        assert result[0].url == f"https://www.tiktok.com/@testuser/video/{first_id}"

    @pytest.mark.asyncio
    async def test_rejects_a_playlist_for_another_creator(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        with (
            patch(
                "tiktok_scraper.scraper._sync_extract",
                return_value={
                    "uploader": "other",
                    "entries": [
                        {
                            "id": "7123456789012345680",
                            "url": "https://www.tiktok.com/@testuser/video/7123456789012345680",
                        }
                    ],
                },
            ),
            pytest.raises(BrowserError, match="different creator"),
        ):
            await _fetch_via_ytdlp(scraper_config)

    def test_ignores_track_channel_and_opaque_uploader_id(self) -> None:
        post_id = "7123456789012345687"
        result = _video_infos_from_info(
            {
                "entries": [
                    {
                        "id": post_id,
                        "url": f"https://www.tiktok.com/@testuser/video/{post_id}",
                        "channel": "Mommy",
                        "uploader": "testuser",
                        "uploader_id": "7162581406329078826",
                    }
                ]
            },
            None,
            "testuser",
        )
        assert [video.video_id for video in result] == [post_id]

    def test_rejects_conflicting_camel_case_original_url(self) -> None:
        post_id = "7123456789012345691"
        result = _video_infos_from_info(
            {
                "entries": [
                    {
                        "id": post_id,
                        "url": f"https://www.tiktok.com/@testuser/video/{post_id}",
                        "originalUrl": f"https://www.tiktok.com/@other/video/{post_id}",
                    }
                ]
            },
            None,
            "testuser",
        )
        assert result == []

    def test_limit_is_applied_after_deduplication_in_source_order(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """A limit selects the first feed records, not the lowest IDs."""
        entries = [
            {
                "id": "7123456789012345699",
                "url": "https://www.tiktok.com/@testuser/video/7123456789012345699",
            },
            {
                "id": "7123456789012345688",
                "url": "https://www.tiktok.com/@testuser/video/7123456789012345688",
            },
            {
                "id": "7123456789012345699",
                "url": "https://www.tiktok.com/@testuser/video/7123456789012345699",
            },
        ]

        result = _video_infos_from_info(
            {"entries": entries},
            2,
            scraper_config.username,
        )

        assert [video.video_id for video in result] == [
            "7123456789012345699",
            "7123456789012345688",
        ]

    def test_explicit_photo_data_overrides_a_video_url_marker(self) -> None:
        post_id = "7123456789012345689"
        result = _video_infos_from_info(
            {
                "entries": [
                    {
                        "id": post_id,
                        "url": f"https://www.tiktok.com/@testuser/video/{post_id}",
                        "uploader": "testuser",
                        "imagePost": {"images": [{"imageURL": {"urlList": []}}]},
                    }
                ]
            },
            None,
            "testuser",
        )
        assert len(result) == 1
        assert result[0].is_photo

    def test_channel_route_requires_the_resolved_channel_id(self) -> None:
        channel_id = (
            "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-"
            "kjp87vNzjY0hG-HZqd"
        )
        other_channel_id = channel_id[:-1] + "x"
        post_id = "7123456789012345690"
        result = _video_infos_from_info(
            {
                "entries": [
                    {
                        "id": post_id,
                        "url": f"https://www.tiktok.com/@{other_channel_id}/video/{post_id}",
                        "uploader": "testuser",
                    }
                ]
            },
            None,
            "testuser",
            allow_opaque_channel_author=True,
            expected_opaque_channel_author=channel_id,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_limit_zero_does_not_call_ytdlp(
        self,
        tmp_path: Path,
    ) -> None:
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            limit=0,
        )
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value={"entries": []},
        ) as mock_extract:
            result = await _fetch_via_ytdlp(config)

        assert result == []
        mock_extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_limit(self, tmp_path: Path) -> None:
        """Should honour the limit setting."""
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            limit=1,
            scroll_delay_min=0.0,
            scroll_delay_max=0.0,
        )
        fake_info = {
            "_type": "playlist",
            "entries": [
                {
                    "id": "7123456789012345678",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345678",
                },
                {
                    "id": "7123456789012345679",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345679",
                },
            ],
        }
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value=fake_info,
        ):
            result = await _fetch_via_ytdlp(config)

        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_sets_playlistend_from_limit(self, tmp_path: Path) -> None:
        """Should pass playlistend when a limit is configured."""
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            limit=7,
            scroll_delay_min=0.0,
            scroll_delay_max=0.0,
        )
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value={"entries": []},
        ) as mock_extract:
            await _fetch_via_ytdlp(config)

        _url, opts = mock_extract.call_args.args
        assert opts["playlistend"] == 7

    @pytest.mark.asyncio
    async def test_enables_impersonation_and_extractor_retries(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Should enable yt-dlp TikTok impersonation and extractor retries."""
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value={"entries": []},
        ) as mock_extract:
            await _fetch_via_ytdlp(scraper_config)

        _url, opts = mock_extract.call_args.args
        assert str(opts["impersonate"]) == "chrome"
        assert opts["extractor_retries"] == max(
            1, min(scraper_config.download_retries, 2)
        )

    @pytest.mark.asyncio
    async def test_deduplicates_videos(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Duplicate entries should be deduplicated."""
        fake_info = {
            "_type": "playlist",
            "entries": [
                {
                    "id": "7123456789012345678",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345678",
                },
                {
                    "id": "7123456789012345678",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345678",
                },
            ],
        }
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value=fake_info,
        ):
            result = await _fetch_via_ytdlp(scraper_config)

        assert len(result) == 1
        assert result[0].video_id == "7123456789012345678"

    @pytest.mark.asyncio
    async def test_empty_when_no_entries(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Should return empty list when no entries are found."""
        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value=None,
        ):
            result = await _fetch_via_ytdlp(scraper_config)

        assert result == []

    @pytest.mark.asyncio
    async def test_creator_not_found(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Should raise CreatorNotFoundError for non-existent users."""
        import yt_dlp

        with (
            patch(
                "tiktok_scraper.scraper._sync_extract",
                side_effect=yt_dlp.utils.DownloadError("This user does not exist"),
            ),
            pytest.raises(CreatorNotFoundError),
        ):
            await _fetch_via_ytdlp(scraper_config)

    @pytest.mark.asyncio
    async def test_materializes_lazy_iterable_entries(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """yt-dlp LazyList-like iterables should be retained and consumed."""

        class LazyEntries:
            def __iter__(self) -> Iterator[dict[str, str]]:
                yield {
                    "id": "7123456789012345678",
                    "url": "https://www.tiktok.com/@testuser/video/7123456789012345678",
                }

        with patch(
            "tiktok_scraper.scraper._sync_extract",
            return_value={"entries": LazyEntries()},
        ):
            result = await _fetch_via_ytdlp(scraper_config)

        assert [video.video_id for video in result] == ["7123456789012345678"]

    def test_generic_not_found_text_is_not_creator_classification(self) -> None:
        assert not _is_creator_not_found("HTTP Error 404: Not Found")
        assert not _is_creator_not_found("Unable to extract video data")
        assert _is_creator_not_found("This user does not exist")

    def test_generic_404_uses_fallback_without_retry(self) -> None:
        import yt_dlp

        error = yt_dlp.utils.DownloadError("HTTP Error 404: Not Found")
        assert not _is_creator_not_found(str(error))
        assert not _is_retryable_ytdlp(error)

    def test_server_errors_are_retryable(self) -> None:
        import yt_dlp

        assert _is_retryable_ytdlp(yt_dlp.utils.DownloadError("HTTP Error 500"))

    def test_sync_extract_materializes_lazy_entries(self) -> None:
        class LazyEntries:
            def __iter__(self) -> Iterator[dict[str, str]]:
                yield {"id": "7123456789012345678"}

        with patch("tiktok_scraper.scraper.yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = {
                "entries": LazyEntries()
            }
            result = _sync_extract("https://example.test", {})

        assert isinstance(result, dict)
        assert result["entries"] == [{"id": "7123456789012345678"}]


class TestEmbedDiscovery:
    """Tests for the embed-based channel-ID discovery fallback."""

    def test_extracts_profile_post_ids_from_frontity_state(self) -> None:
        html = _frontity_html(
            "/embed/@testuser",
            {
                "videoList": [
                    {"id": "7123456789012345678", "author": "testuser"},
                    {"id": "7123456789012345679"},
                    {"id": "7123456789012345680", "author": "other"},
                    {"id": "1234", "author": "testuser"},
                ]
            },
        )

        assert _extract_embed_profile_post_ids(html, "testuser") == [
            "7123456789012345678",
            "7123456789012345679",
        ]

    def test_profile_embed_ignores_external_post_links(self) -> None:
        html = (
            '<a href="https://example.com/@testuser/video/7123456789012345680">bad</a>'
            '<a href="//www.tiktok.com/@testuser/video/7123456789012345681">good</a>'
        )

        assert _extract_embed_profile_post_ids(html, "testuser") == [
            "7123456789012345681"
        ]

    def test_extracts_channel_id_from_video_embed(self) -> None:
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {"videoData": {"authorInfos": {"secUid": channel_id}}},
        )

        assert _extract_embed_channel_id(html, "7123456789012345678") == channel_id

    def test_channel_id_comes_only_from_author_data(self) -> None:
        decoy = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-DEC"
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {
                "music": {"secUid": decoy},
                "videoData": {"authorInfos": {"secUid": channel_id}},
            },
        )

        assert _extract_embed_channel_id(html, "7123456789012345678") == channel_id

    @pytest.mark.asyncio
    async def test_resolves_channel_id_and_uses_ytdlp_channel_route(
        self,
        tmp_path: Path,
    ) -> None:
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            request_delay=0.0,
        )
        profile_html = _frontity_html(
            "/embed/@testuser",
            {"videoList": [{"id": "7123456789012345678"}]},
        )
        video_html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {
                "videoData": {
                    "authorInfos": {
                        "uniqueId": "testuser",
                        "secUid": channel_id,
                    }
                }
            },
        )
        expected = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            )
        ]

        with (
            patch(
                "tiktok_scraper.scraper._sync_fetch_embed_html",
                side_effect=[profile_html, video_html],
            ) as mock_fetch,
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp_channel",
                new=AsyncMock(return_value=expected),
            ) as mock_channel,
        ):
            result = await _fetch_via_embed_channel(config)

        assert result == expected
        assert mock_fetch.call_count == 2
        mock_channel.assert_awaited_once_with(config, channel_id)

    @pytest.mark.asyncio
    async def test_profile_discovery_falls_back_to_embed_channel_route(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        import yt_dlp

        expected = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            )
        ]
        with (
            patch(
                "tiktok_scraper.scraper._sync_extract",
                side_effect=yt_dlp.utils.DownloadError(
                    "Unable to extract secondary user ID"
                ),
            ),
            patch(
                "tiktok_scraper.scraper._fetch_via_embed_channel",
                new=AsyncMock(return_value=expected),
            ) as mock_embed,
        ):
            result = await _fetch_via_ytdlp(scraper_config)

        assert result == expected
        mock_embed.assert_awaited_once_with(scraper_config)

    @pytest.mark.asyncio
    async def test_unverified_cached_post_is_not_used_for_channel_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            request_delay=0.0,
        )
        config.output_dir.mkdir()
        (config.output_dir / "7123456789012345678.mp4").write_bytes(
            b"\x00\x00\x00\x18ftypmp42"
        )
        video_html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {"videoData": {"authorInfos": {"secUid": channel_id}}},
        )

        with (
            patch(
                "tiktok_scraper.scraper._sync_fetch_embed_html",
                side_effect=[RuntimeError("profile embed unavailable"), video_html],
            ) as mock_fetch,
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp_channel",
                new=AsyncMock(return_value=[]),
            ) as mock_channel,
        ):
            await _fetch_via_embed_channel(config)

        assert mock_fetch.call_count == 2
        mock_channel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cached_post_requires_matching_explicit_author(
        self,
        tmp_path: Path,
    ) -> None:
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            request_delay=0.0,
        )
        config.output_dir.mkdir()
        (config.output_dir / "7123456789012345678.mp4").write_bytes(
            b"\x00\x00\x00\x18ftypmp42"
        )
        video_html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {
                "videoData": {
                    "authorInfos": {
                        "uniqueId": "testuser",
                        "secUid": channel_id,
                    }
                }
            },
        )

        with (
            patch(
                "tiktok_scraper.scraper._sync_fetch_embed_html",
                side_effect=[RuntimeError("profile embed unavailable"), video_html],
            ),
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp_channel",
                new=AsyncMock(return_value=[]),
            ) as mock_channel,
        ):
            await _fetch_via_embed_channel(config)

        mock_channel.assert_awaited_once_with(config, channel_id)

    @pytest.mark.asyncio
    async def test_embed_probe_rejects_a_different_explicit_author(
        self,
        tmp_path: Path,
    ) -> None:
        channel_id = "MS4wLjABAAAAX2WwIM9zaUvOw_9bZbYJeGD7AAlYmxEkHfP7SNKrO505N-kjp87vNzjY0hG-HZqd"
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            request_delay=0.0,
        )
        profile_html = _frontity_html(
            "/embed/@testuser",
            {"videoList": [{"id": "7123456789012345678"}]},
        )
        video_html = _frontity_html(
            "/embed/v2/7123456789012345678",
            {
                "videoData": {
                    "authorInfos": {
                        "uniqueId": "other",
                        "secUid": channel_id,
                    }
                }
            },
        )

        with (
            patch(
                "tiktok_scraper.scraper._sync_fetch_embed_html",
                side_effect=[profile_html, video_html],
            ),
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp_channel",
                new=AsyncMock(return_value=[]),
            ) as mock_channel,
        ):
            assert await _fetch_via_embed_channel(config) == []

        mock_channel.assert_not_awaited()


def test_access_marker_in_user_content_is_not_a_block_page() -> None:
    html = (
        '<a href="/@testuser/video/7123456789012345678">'
        "login to continue reading this caption"
        "</a>"
    )
    assert not _html_indicates_inconclusive(html)


def test_short_login_wall_is_inconclusive() -> None:
    assert _html_indicates_inconclusive("<h1>Log in to continue</h1>")


def test_login_username_is_not_an_access_wall_title() -> None:
    assert not _title_indicates_inconclusive("login | TikTok")
    assert _title_indicates_inconclusive("Log in to continue | TikTok")


def test_ordinary_captcha_asset_class_is_not_an_access_wall() -> None:
    assert not _html_indicates_inconclusive('<div class="captcha-icon"></div>')
    assert _html_indicates_inconclusive('<div data-e2e="captcha-verify"></div>')


class TestApiInterception:
    """Tests for current and legacy TikTok item-list endpoints."""

    @pytest.mark.asyncio
    async def test_accepts_creator_item_list_endpoint(self) -> None:
        videos_by_id: dict[str, VideoInfo] = {}
        page = MagicMock()
        _attach_api_interceptor(page, videos_by_id)
        callback = page.on.call_args.args[1]
        response = AsyncMock()
        response.url = "https://www.tiktok.com/api/creator/item_list/"
        response.body.return_value = json.dumps(
            {
                "itemList": [
                    {"id": "7123456789012345678", "author": {"uniqueId": "testuser"}}
                ]
            }
        ).encode()

        await callback(response)

        assert videos_by_id["7123456789012345678"].url.endswith(
            "/@testuser/video/7123456789012345678"
        )

    @pytest.mark.asyncio
    async def test_rejects_wrong_authors_external_links_and_bad_ids(self) -> None:
        videos_by_id: dict[str, VideoInfo] = {}
        page = MagicMock()
        _attach_api_interceptor(page, videos_by_id, "testuser")
        callback = page.on.call_args.args[1]
        response = AsyncMock()
        response.url = "https://www.tiktok.com/api/post/item_list/"
        response.status = 200
        response.body.return_value = json.dumps(
            {
                "itemList": [
                    {
                        "id": "7123456789012345680",
                        "author": {"uniqueId": "testuser"},
                    },
                    {
                        "id": "7123456789012345681",
                        "author": {"uniqueId": "other"},
                    },
                    {
                        "id": "7123456789012345682",
                        "url": "https://example.com/@testuser/video/7123456789012345682",
                        "author": {"uniqueId": "testuser"},
                    },
                    {
                        "id": "7123",
                        "author": {"uniqueId": "testuser"},
                    },
                ]
            }
        ).encode()

        await callback(response)

        assert list(videos_by_id) == ["7123456789012345680"]

    @pytest.mark.asyncio
    async def test_callback_task_is_awaited_by_browser(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        listeners: list[Callable[[object], object]] = []
        page.on.side_effect = lambda _event, callback: listeners.append(callback)

        response = AsyncMock()
        response.url = "https://www.tiktok.com/api/post/item_list/"
        response.status = 200

        async def delayed_body() -> bytes:
            await asyncio.sleep(0.01)
            return json.dumps(
                {
                    "itemList": [
                        {
                            "id": "7123456789012345680",
                            "author": {"uniqueId": "testuser"},
                        }
                    ]
                }
            ).encode()

        response.body.side_effect = delayed_body

        async def goto(_url: str, *, wait_until: str, timeout: int) -> AsyncMock:
            assert wait_until == "domcontentloaded"
            assert timeout == 30000
            assert listeners
            listeners[0](response)
            return response

        page.goto.side_effect = goto

        result = await _fetch_via_browser(scraper_config, mock_browser_context)

        assert [video.video_id for video in result] == ["7123456789012345680"]


class TestHydrationDiscovery:
    """Tests for global and script-element hydration payloads."""

    @pytest.mark.asyncio
    async def test_reads_hydration_script_elements_and_filters_authors(self) -> None:
        page = AsyncMock()
        page.evaluate.return_value = None
        script = AsyncMock()
        script.text_content.return_value = json.dumps(
            {
                "__DEFAULT_SCOPE__": {
                    "webapp.user-post": {
                        "itemList": [
                            {
                                "id": "7123456789012345680",
                                "author": {"uniqueId": "testuser"},
                            },
                            {
                                "id": "7123456789012345681",
                                "author": {"uniqueId": "other"},
                            },
                        ]
                    }
                }
            }
        )
        page.query_selector_all.return_value = [script]
        videos_by_id: dict[str, VideoInfo] = {}

        await _extract_hydration_data(page, videos_by_id, "testuser")

        assert list(videos_by_id) == ["7123456789012345680"]
        assert videos_by_id["7123456789012345680"].url == (
            "https://www.tiktok.com/@testuser/video/7123456789012345680"
        )

    def test_hydration_item_module_rejects_malformed_records(self) -> None:
        videos_by_id: dict[str, VideoInfo] = {}

        _parse_hydration_items(
            {
                "ItemModule": {
                    "7123456789012345680": {"author": "testuser"},
                    "7123456789012345681": {"author": "other"},
                    "bad": {"author": "testuser"},
                }
            },
            videos_by_id,
            "testuser",
        )

        assert list(videos_by_id) == ["7123456789012345680"]


class TestFetchViaBrowser:
    """Tests for the Playwright browser fallback strategy."""

    @pytest.mark.asyncio
    async def test_creator_not_found_via_status_code(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        """A 404 response should raise CreatorNotFoundError."""
        page = mock_browser_context.new_page.return_value
        response = AsyncMock()
        response.status = 404
        page.goto.return_value = response

        with pytest.raises(CreatorNotFoundError):
            await _fetch_via_browser(scraper_config, mock_browser_context)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [403, 429, 500, 502, 503, 504])
    async def test_blocking_status_is_not_an_empty_success(
        self,
        status: int,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        page.goto.return_value.status = status

        with pytest.raises(BrowserError, match="inconclusive"):
            await _fetch_via_browser(scraper_config, mock_browser_context)

        page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creator_not_found_via_dom(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        """DOM indicator for missing account should raise CreatorNotFoundError."""
        page = mock_browser_context.new_page.return_value

        async def _query_selector(selector: str) -> MagicMock | None:
            if "not-found" in selector or "Couldn't find" in selector:
                return MagicMock()
            return None

        page.query_selector.side_effect = _query_selector

        with pytest.raises(CreatorNotFoundError):
            await _fetch_via_browser(scraper_config, mock_browser_context)

    @pytest.mark.asyncio
    async def test_collects_videos_from_dom(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        """Should collect video URLs from DOM links."""
        page = mock_browser_context.new_page.return_value

        video_call_count = 0

        async def _query_selector_all(selector: str) -> list[AsyncMock]:
            nonlocal video_call_count
            if "/video/" not in selector:
                return []
            if video_call_count == 0:
                video_call_count += 1
                link1 = AsyncMock()
                link1.get_attribute.return_value = (
                    "https://www.tiktok.com/@testuser/video/7123456789012345678"
                )
                link2 = AsyncMock()
                link2.get_attribute.return_value = (
                    "https://www.tiktok.com/@testuser/video/7123456789012345679"
                )
                return [link1, link2]
            return []

        page.query_selector_all.side_effect = _query_selector_all

        result = await _fetch_via_browser(scraper_config, mock_browser_context)

        assert len(result) == 2
        ids = {v.video_id for v in result}
        assert "7123456789012345678" in ids
        assert "7123456789012345679" in ids
        page.wait_for_function.assert_awaited_once()
        page.wait_for_timeout.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dom_rejects_external_and_wrong_creator_links(self) -> None:
        page = AsyncMock()
        links = []
        for href in (
            "https://www.tiktok.com/@testuser/video/7123456789012345680?lang=en",
            "https://www.tiktok.com/@other/video/7123456789012345681",
            "https://example.com/@testuser/video/7123456789012345682",
            "/@testuser/video/7123456789012345683",
        ):
            link = AsyncMock()
            link.get_attribute.return_value = href
            links.append(link)
        page.query_selector_all.return_value = links
        videos_by_id: dict[str, VideoInfo] = {}

        await _collect_from_dom(page, videos_by_id, "testuser")

        assert list(videos_by_id) == [
            "7123456789012345680",
            "7123456789012345683",
        ]
        assert videos_by_id["7123456789012345680"].url == (
            "https://www.tiktok.com/@testuser/video/7123456789012345680"
        )

    @pytest.mark.asyncio
    async def test_valid_empty_profile_is_allowed(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        page.title.return_value = "testuser | TikTok"
        page.content.return_value = "<html><body>testuser</body></html>"

        assert await _fetch_via_browser(scraper_config, mock_browser_context) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "title",
        [
            "Log in | TikTok",
            "Private account | TikTok",
            "Verify you are human | TikTok",
        ],
    )
    async def test_gated_profile_is_not_an_empty_success(
        self,
        title: str,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        page.title.return_value = title

        with pytest.raises(BrowserError, match="inconclusive"):
            await _fetch_via_browser(scraper_config, mock_browser_context)

    @pytest.mark.asyncio
    async def test_numeric_404_title_does_not_imply_missing_creator(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        page.title.return_value = "404 | TikTok"

        with pytest.raises(BrowserError, match="inconclusive"):
            await _fetch_via_browser(scraper_config, mock_browser_context)

    @pytest.mark.asyncio
    async def test_stops_after_stale_scrolls(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        """Should stop scrolling after MAX_STALE_SCROLLS with no new content."""
        page = mock_browser_context.new_page.return_value
        page.query_selector_all.return_value = []

        result = await _fetch_via_browser(scraper_config, mock_browser_context)

        assert len(result) == 0
        assert page.evaluate.call_count > 0


class TestFetchVideoUrls:
    """Tests for the top-level fetch_video_urls orchestrator."""

    @pytest.mark.asyncio
    async def test_limit_zero_skips_all_network_setup(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            username="testuser",
            output_dir=tmp_path / "downloads",
            limit=0,
        )
        with (
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp",
                new=AsyncMock(),
            ) as mock_ytdlp,
            patch(
                "tiktok_scraper.scraper.create_stealth_browser_context",
                new=AsyncMock(),
            ) as mock_browser,
        ):
            assert await fetch_video_urls(config) == []

        mock_ytdlp.assert_not_awaited()
        mock_browser.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_ytdlp_primary(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """Should use yt-dlp as the primary discovery method."""
        expected = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]
        with patch(
            "tiktok_scraper.scraper._fetch_via_ytdlp",
            return_value=expected,
        ) as mock_ytdlp:
            result = await fetch_video_urls(scraper_config, browser_context=None)

        mock_ytdlp.assert_awaited_once_with(scraper_config)
        assert result == expected

    @pytest.mark.asyncio
    async def test_falls_back_to_browser(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        """Should fall back to browser when yt-dlp fails."""
        expected = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]
        with (
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp",
                side_effect=RuntimeError("yt-dlp failed"),
            ),
            patch(
                "tiktok_scraper.scraper._fetch_via_browser",
                return_value=expected,
            ) as mock_browser,
        ):
            result = await fetch_video_urls(
                scraper_config,
                browser_context=mock_browser_context,
            )

        mock_browser.assert_awaited_once_with(scraper_config, mock_browser_context)
        assert result == expected

    @pytest.mark.asyncio
    async def test_creates_browser_for_fallback_when_none_is_supplied(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        """The public discovery path should start and clean up its own fallback."""
        expected = [
            VideoInfo(
                video_id="7123456789012345678",
                url="https://www.tiktok.com/@testuser/video/7123456789012345678",
            ),
        ]
        playwright = AsyncMock()
        browser = AsyncMock()
        context = AsyncMock()

        with (
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp",
                side_effect=RuntimeError("yt-dlp failed"),
            ),
            patch(
                "tiktok_scraper.scraper.create_stealth_browser_context",
                return_value=(playwright, browser, context),
            ) as mock_create_context,
            patch(
                "tiktok_scraper.scraper._fetch_via_browser",
                return_value=expected,
            ) as mock_browser_fetch,
        ):
            result = await fetch_video_urls(scraper_config)

        assert result == expected
        mock_create_context.assert_awaited_once_with(
            headless=scraper_config.headless,
            cookies_file=scraper_config.cookies_file,
        )
        mock_browser_fetch.assert_awaited_once_with(scraper_config, context)
        context.close.assert_awaited_once()
        browser.close.assert_awaited_once()
        playwright.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_page_cleanup_does_not_replace_primary_error(
        self,
        scraper_config: ScraperConfig,
        mock_browser_context: AsyncMock,
    ) -> None:
        page = mock_browser_context.new_page.return_value
        page.goto.side_effect = RuntimeError(
            "primary failure https://www.tiktok.com/x?token=secret"
        )
        page.close.side_effect = RuntimeError("cleanup failure")

        with pytest.raises(BrowserError, match="primary failure") as caught:
            await _fetch_via_browser(scraper_config, mock_browser_context)

        assert "secret" not in str(caught.value)
        page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owned_resource_cleanup_continues_after_one_failure(
        self,
        scraper_config: ScraperConfig,
    ) -> None:
        playwright = AsyncMock()
        browser = AsyncMock()
        context = AsyncMock()
        context.close.side_effect = RuntimeError("context cleanup failure")

        with (
            patch(
                "tiktok_scraper.scraper._fetch_via_ytdlp",
                side_effect=RuntimeError("yt-dlp failed"),
            ),
            patch(
                "tiktok_scraper.scraper.create_stealth_browser_context",
                return_value=(playwright, browser, context),
            ),
            patch(
                "tiktok_scraper.scraper._fetch_via_browser",
                side_effect=BrowserError("primary browser failure"),
            ),
            pytest.raises(BrowserError, match="primary browser failure"),
        ):
            await fetch_video_urls(scraper_config)

        context.close.assert_awaited_once()
        browser.close.assert_awaited_once()
        playwright.stop.assert_awaited_once()

    def test_error_log_redaction_removes_query_and_cookie_values(self) -> None:
        redacted = _redact_log_value(
            "request https://www.tiktok.com/post?id=post-secret&token=token-secret "
            "cookiefile=/private/cookies.txt"
        )

        assert "post-secret" not in redacted
        assert "token-secret" not in redacted
        assert "/private/cookies.txt" not in redacted
