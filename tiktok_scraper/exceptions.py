"""Custom exception hierarchy for the TikTok scraper."""


class ScraperError(Exception):
    """Base exception for all scraper errors."""


class CreatorNotFoundError(ScraperError):
    """Raised when the TikTok username does not exist."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"TikTok creator @{username} not found.")


class RateLimitError(ScraperError):
    """Raised when TikTok rate-limits the request."""

    def __init__(self, message: str = "Rate limited by TikTok.") -> None:
        super().__init__(message)


class DownloadError(ScraperError):
    """Raised when a video download fails."""

    def __init__(self, video_id: str, reason: str) -> None:
        self.video_id = video_id
        self.reason = reason
        super().__init__(f"Failed to download video {video_id}: {reason}")


class BrowserError(ScraperError):
    """Raised when browser automation encounters an unrecoverable error."""

    def __init__(self, message: str = "Browser automation failed.") -> None:
        super().__init__(message)


class FFmpegError(ScraperError):
    """Raised when a download needs ffmpeg but it is unavailable."""

    def __init__(self, message: str = "ffmpeg is required for this download.") -> None:
        super().__init__(message)
