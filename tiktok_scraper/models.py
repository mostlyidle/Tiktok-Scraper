"""Data classes for videos and scrape results."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoInfo:
    """Represents a discovered TikTok video."""

    video_id: str
    url: str


@dataclass
class ScrapeResult:
    """Summary of a scraping session."""

    username: str
    total_found: int = 0
    downloaded: int = 0
    photos_downloaded: int = 0
    skipped: int = 0
    skipped_audio_only: int = 0
    failed: int = 0
    failed_videos: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate download success rate as a percentage."""
        attempted = self.downloaded + self.photos_downloaded + self.failed
        if attempted == 0:
            return 0.0
        return ((self.downloaded + self.photos_downloaded) / attempted) * 100.0
