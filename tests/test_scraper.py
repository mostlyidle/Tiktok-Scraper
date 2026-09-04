"""Unit tests for the scraper module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.exceptions import CreatorNotFoundError
from tiktok_scraper.models import VideoInfo
from tiktok_scraper.scraper import (
    _fetch_via_browser,
    _fetch_via_ytdlp,
    _is_creator_not_found,
    _is_retryable_ytdlp,
    _sync_extract,
    fetch_video_urls,
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
        assert opts["extractor_retries"] == scraper_config.download_retries

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
            def __iter__(self):
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

    def test_sync_extract_materializes_lazy_entries(self) -> None:
        class LazyEntries:
            def __iter__(self):
                yield {"id": "7123456789012345678"}

        with patch("tiktok_scraper.scraper.yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.extract_info.return_value = {
                "entries": LazyEntries()
            }
            result = _sync_extract("https://example.test", {})

        assert isinstance(result, dict)
        assert result["entries"] == [{"id": "7123456789012345678"}]


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
