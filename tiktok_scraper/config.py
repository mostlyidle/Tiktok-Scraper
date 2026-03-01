"""Configuration and environment variables."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScraperConfig:
    """Configuration for a scraping session."""

    username: str
    output_dir: Path
    cookies_file: Path | None = None
    limit: int | None = None
    headless: bool = True
    max_concurrent_downloads: int = 3
    scroll_delay_min: float = 2.0
    scroll_delay_max: float = 5.0
    download_retries: int = 3
    video_only: bool = False

    @staticmethod
    def default_output_dir(username: str) -> Path:
        """Return the default output directory for a given username."""
        return Path("./downloads") / username
