"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tiktok_scraper.config import ScraperConfig
from tiktok_scraper.models import VideoInfo


@pytest.fixture()
def sample_video_urls() -> list[str]:
    """A list of fake TikTok video URLs for testing."""
    return [
        "https://www.tiktok.com/@testuser/video/7123456789012345678",
        "https://www.tiktok.com/@testuser/video/7123456789012345679",
        "https://www.tiktok.com/@testuser/video/7123456789012345680",
        "https://www.tiktok.com/@testuser/video/7123456789012345681",
        "https://www.tiktok.com/@testuser/video/7123456789012345682",
    ]


@pytest.fixture()
def sample_videos() -> list[VideoInfo]:
    """A list of :class:`VideoInfo` objects for testing."""
    return [
        VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@testuser/video/7123456789012345678",
        ),
        VideoInfo(
            video_id="7123456789012345679",
            url="https://www.tiktok.com/@testuser/video/7123456789012345679",
        ),
        VideoInfo(
            video_id="7123456789012345680",
            url="https://www.tiktok.com/@testuser/video/7123456789012345680",
        ),
    ]


@pytest.fixture()
def scraper_config(tmp_path: Path) -> ScraperConfig:
    """A :class:`ScraperConfig` pointed at a temporary directory."""
    return ScraperConfig(
        username="testuser",
        output_dir=tmp_path / "downloads",
        scroll_delay_min=0.0,
        scroll_delay_max=0.0,
    )


@pytest.fixture()
def mock_browser_context() -> AsyncMock:
    """A mocked Playwright browser context."""
    context = AsyncMock()
    page = AsyncMock()
    context.new_page.return_value = page

    # Default: successful navigation with status 200.
    response = AsyncMock()
    response.status = 200
    page.goto.return_value = response

    # Default: no "not found" indicators.
    page.query_selector.return_value = None

    # Default: empty video link list.
    page.query_selector_all.return_value = []

    # Default: page title for a valid creator.
    page.title.return_value = "testuser | TikTok"

    # Default: no hydration data.
    page.evaluate.return_value = None

    # page.on is a synchronous registration method — use MagicMock to
    # avoid unawaited-coroutine warnings from AsyncMock.
    page.on = MagicMock()

    return context


@pytest.fixture()
def video_only_config(tmp_path: Path) -> ScraperConfig:
    """A :class:`ScraperConfig` with ``video_only=True``."""
    return ScraperConfig(
        username="testuser",
        output_dir=tmp_path / "downloads",
        scroll_delay_min=0.0,
        scroll_delay_max=0.0,
        video_only=True,
    )
