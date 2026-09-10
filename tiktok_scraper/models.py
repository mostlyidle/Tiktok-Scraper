"""Data classes for videos and scrape results."""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_VIDEO_ID_PATTERN = re.compile(r"\d{15,25}\Z")
_POST_PATH_PATTERN = re.compile(
    r"^/@(?P<author>[A-Za-z0-9._]{1,24})/"
    r"(?P<kind>video|photo)/(?P<id>\d{15,25})/?\Z",
    re.IGNORECASE,
)
_TIKTOK_HOSTS = frozenset(
    {
        "tiktok.com",
        "www.tiktok.com",
        "m.tiktok.com",
        "tiktokv.com",
        "www.tiktokv.com",
    }
)


def _validate_video_url(url: str, video_id: str) -> None:
    """Validate a TikTok post URL and its relationship to *video_id*."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty TikTok post URL")

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("url must be a valid TikTok post URL") from exc

    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("url must be a valid TikTok post URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or hostname.lower() not in _TIKTOK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
    ):
        raise ValueError("url must point to a TikTok host")

    match = _POST_PATH_PATTERN.fullmatch(parsed.path)
    if match is None or match.group("id") != video_id:
        raise ValueError("url path must contain the matching video ID")
    author = match.group("author")
    if author in {".", ".."} or author.startswith(".") or author.endswith("."):
        raise ValueError("url path must contain a valid TikTok creator")


@dataclass(frozen=True)
class VideoInfo:
    """Represents a discovered TikTok video."""

    video_id: str
    url: str

    def __post_init__(self) -> None:
        """Reject malformed IDs and URLs before they reach the downloader."""
        if not isinstance(self.video_id, str) or not _VIDEO_ID_PATTERN.fullmatch(
            self.video_id
        ):
            raise ValueError("video_id must contain 15 to 25 decimal digits")
        _validate_video_url(self.url, self.video_id)

    @property
    def username(self) -> str:
        """Return the creator handle encoded in the canonical post URL."""
        match = _POST_PATH_PATTERN.fullmatch(urlsplit(self.url).path)
        if match is None:  # pragma: no cover - enforced by __post_init__
            raise ValueError("url is not a valid TikTok post URL")
        return match.group("author")

    @property
    def is_photo(self) -> bool:
        """Return whether the post URL identifies a photo slideshow."""
        match = _POST_PATH_PATTERN.fullmatch(urlsplit(self.url).path)
        return match is not None and match.group("kind").casefold() == "photo"


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
    audio_downloaded: int = 0
    failed_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate download success rate as a percentage."""
        attempted = (
            self.downloaded
            + self.audio_downloaded
            + self.photos_downloaded
            + self.failed
        )
        if attempted == 0:
            return 0.0
        return (
            (self.downloaded + self.audio_downloaded + self.photos_downloaded)
            / attempted
        ) * 100.0
