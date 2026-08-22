"""Tests for validation, resume, and retry utility helpers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pytest

from tiktok_scraper.config import ScraperConfig, normalize_username
from tiktok_scraper.utils import (
    bounded_backoff,
    cleanup_temporary_artifacts,
    get_downloaded_ids,
    is_completed_artifact,
    parse_retry_after,
)


def test_username_rejects_dot_paths() -> None:
    with pytest.raises(ValueError):
        normalize_username(".")
    with pytest.raises(ValueError):
        normalize_username("..")


def test_default_output_is_under_download_root() -> None:
    config = ScraperConfig(username="@safe_user", output_dir=Path("downloads"))
    assert config.username == "safe_user"
    assert ScraperConfig.default_output_dir("safe_user") == Path("downloads/safe_user")


@pytest.mark.parametrize(
    "field,value",
    [
        ("limit", -1),
        ("max_concurrent_downloads", 0),
        ("download_retries", -1),
        ("scroll_delay_min", -1.0),
        ("request_delay", -1.0),
        ("retry_backoff_base", 0.0),
        ("retry_backoff_max", 0.5),
    ],
)
def test_config_rejects_invalid_invariants(field: str, value: int | float) -> None:
    kwargs: dict[str, object] = {field: value}
    if field == "retry_backoff_max":
        kwargs["retry_backoff_base"] = 1.0
    with pytest.raises(ValueError):
        ScraperConfig(username="testuser", output_dir=Path("downloads"), **kwargs)  # type: ignore[arg-type]


def test_resume_requires_nonempty_completed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "downloads"
    output.mkdir()
    (output / "7123456789012345678.mp4").touch()
    (output / "7123456789012345679.mp4.part").write_bytes(b"partial")
    empty_dir = output / "7123456789012345680"
    empty_dir.mkdir()
    complete_dir = output / "7123456789012345681"
    complete_dir.mkdir()
    (complete_dir / "image_1.jpg").write_bytes(b"image")

    assert get_downloaded_ids(output) == {"7123456789012345681"}
    assert not is_completed_artifact(output / "7123456789012345678.mp4")
    assert is_completed_artifact(complete_dir)


def test_cleanup_only_removes_slideshow_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "downloads"
    output.mkdir()
    root_part = output / "7123456789012345678.mp4.part"
    root_ytdl = output / "7123456789012345678.mp4.ytdl"
    root_part.write_bytes(b"resume")
    root_ytdl.write_bytes(b"resume")
    staging = output / ".7123456789012345678.partial"
    staging.mkdir()
    stale = staging / "image_1.jpg.part"
    stale.write_bytes(b"stale")
    complete = staging / "image_2.jpg"
    complete.write_bytes(b"keep")

    cleanup_temporary_artifacts(output)

    assert root_part.exists()
    assert root_ytdl.exists()
    assert not stale.exists()
    assert complete.exists()


def test_retry_after_parses_seconds_and_http_dates() -> None:
    assert parse_retry_after("3.5") == 3.5
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.UTC)
    assert parse_retry_after("Wed, 19 Aug 2026 12:00:05 GMT", now=now) == 5.0
    assert parse_retry_after("not-a-date") is None


def test_backoff_is_jittered_and_capped() -> None:
    with patch("tiktok_scraper.utils.random.uniform", return_value=1.0):
        assert bounded_backoff(4, base=1, cap=2) == 2
