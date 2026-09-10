"""Validated runtime configuration for a scraping session."""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,24}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)


def normalize_username(username: str) -> str:
    """Validate and normalise a TikTok handle.

    Handles are kept deliberately conservative because the value is used in
    both URLs and the default output path.  A leading ``@`` is accepted for
    CLI convenience, but is never stored in the configuration.
    """
    normalized = username.strip().removeprefix("@")
    if (
        normalized in {".", ".."}
        or not normalized
        or not _USERNAME_PATTERN.fullmatch(normalized)
        or normalized.startswith(".")
        or normalized.endswith(".")
    ):
        raise ValueError(
            "username must contain only letters, numbers, '.', or '_' "
            "and be at most 24 characters"
        )
    return normalized


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
    request_delay: float = 1.0
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 60.0
    request_timeout: float = 30.0
    image_max_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject configurations that could cause invalid or unsafe runs."""
        object.__setattr__(self, "username", normalize_username(self.username))
        output_dir = Path(self.output_dir)
        object.__setattr__(self, "output_dir", output_dir)
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            raise ValueError("output_dir must be a directory or a new directory path")
        if self.cookies_file is not None:
            cookies_file = Path(self.cookies_file)
            object.__setattr__(self, "cookies_file", cookies_file)
            if cookies_file.is_symlink() or not cookies_file.is_file():
                raise ValueError("cookies_file must be a regular file")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.max_concurrent_downloads < 1:
            raise ValueError("max_concurrent_downloads must be at least 1")
        if self.download_retries < 0:
            raise ValueError("download_retries must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.scroll_delay_min,
                self.scroll_delay_max,
                self.request_delay,
                self.retry_backoff_base,
                self.retry_backoff_max,
                self.request_timeout,
            )
        ):
            raise ValueError("delay and backoff values must be finite")
        if self.scroll_delay_min < 0 or self.scroll_delay_max < 0:
            raise ValueError("scroll delays must be non-negative")
        if self.scroll_delay_min > self.scroll_delay_max:
            raise ValueError("scroll_delay_min must not exceed scroll_delay_max")
        if self.request_delay < 0:
            raise ValueError("request_delay must be non-negative")
        if self.retry_backoff_base <= 0:
            raise ValueError("retry_backoff_base must be positive")
        if self.retry_backoff_max < self.retry_backoff_base:
            raise ValueError("retry_backoff_max must not be below retry_backoff_base")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.image_max_bytes < 1:
            raise ValueError("image_max_bytes must be positive")

    @staticmethod
    def default_output_dir(username: str) -> Path:
        """Return the default output directory for a given username."""
        root = Path("./downloads")
        normalized = normalize_username(username)
        directory_name = normalized
        if (
            os.name == "nt"
            and normalized.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            directory_name = f"_{normalized}"
        candidate = root / directory_name
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError("username would escape the downloads directory")
        return candidate
