"""Tests for validation, resume, and retry utility helpers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import tiktok_scraper.utils as utils
from tiktok_scraper.config import ScraperConfig, normalize_username
from tiktok_scraper.models import VideoInfo
from tiktok_scraper.utils import (
    RequestLimiter,
    bounded_backoff,
    cleanup_temporary_artifacts,
    extract_post_id,
    extract_video_id,
    get_downloaded_ids,
    is_completed_artifact,
    is_completed_slideshow_artifact,
    isolated_cookie_file,
    netscape_cookie_jar,
    netscape_cookies_for_playwright,
    parse_netscape_cookies,
    parse_retry_after,
    read_netscape_cookie_pairs,
    safe_output_path,
    validate_output_dir,
)

_VALID_JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
_VALID_MP4 = b"\x00\x00\x00\x18ftypisomfake-mp4"


def test_username_rejects_dot_paths() -> None:
    with pytest.raises(ValueError):
        normalize_username(".")
    with pytest.raises(ValueError):
        normalize_username("..")
    with pytest.raises(ValueError):
        normalize_username(".username")
    with pytest.raises(ValueError):
        normalize_username("username.")


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


@pytest.mark.parametrize(
    "field",
    [
        "scroll_delay_min",
        "scroll_delay_max",
        "request_delay",
        "retry_backoff_base",
        "retry_backoff_max",
        "request_timeout",
    ],
)
def test_config_rejects_nonfinite_delays(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScraperConfig(
            username="testuser",
            output_dir=Path("downloads"),
            **{field: float("nan")},
        )  # type: ignore[arg-type]


def test_resume_requires_nonempty_completed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "downloads"
    output.mkdir()
    (output / "7123456789012345678.mp4").write_bytes(b"not media")
    (output / "7123456789012345679.mp4.part").write_bytes(b"partial")
    empty_dir = output / "7123456789012345680"
    empty_dir.mkdir()
    complete_dir = output / "7123456789012345681"
    complete_dir.mkdir()
    (complete_dir / "image_1.jpg").write_bytes(_VALID_JPEG)

    assert get_downloaded_ids(output) == {"7123456789012345681"}
    assert not is_completed_artifact(output / "7123456789012345678.mp4")
    assert is_completed_artifact(complete_dir)
    assert is_completed_slideshow_artifact(complete_dir)


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
    download_staging = output / ".7123456789012345679.download"
    download_staging.mkdir()
    (download_staging / "7123456789012345679.mp4.part").write_bytes(b"stale")

    cleanup_temporary_artifacts(output)

    assert root_part.exists()
    assert root_ytdl.exists()
    assert not staging.exists()
    assert not download_staging.exists()


def test_video_info_rejects_invalid_ids_and_mismatched_urls() -> None:
    with pytest.raises(ValueError):
        VideoInfo(
            video_id="short",
            url="https://www.tiktok.com/@user/video/7123456789012345678",
        )
    with pytest.raises(ValueError):
        VideoInfo(
            video_id="7123456789012345678",
            url="https://example.com/@user/video/7123456789012345678",
        )
    with pytest.raises(ValueError):
        VideoInfo(
            video_id="7123456789012345678",
            url="https://www.tiktok.com/@user/photo/7123456789012345679",
        )
    with pytest.raises(ValueError):
        VideoInfo(
            video_id="7123456789012345678",
            url="http://www.tiktok.com/@user/video/7123456789012345678",
        )


def test_video_info_accepts_video_and_photo_urls() -> None:
    video_id = "7123456789012345678"
    assert VideoInfo(
        video_id=video_id,
        url=f"https://www.tiktok.com/@user/video/{video_id}?lang=en",
    ).video_id == video_id
    assert VideoInfo(
        video_id=video_id,
        url=f"https://www.tiktok.com/@user/photo/{video_id}/",
    ).url.endswith("/")


def test_official_tiktokv_post_urls_remain_supported() -> None:
    video_id = "7123456789012345678"
    url = f"https://tiktokv.com/@user/video/{video_id}"
    assert VideoInfo(video_id=video_id, url=url).username == "user"
    assert extract_video_id(url) == video_id
    www_url = f"https://www.tiktokv.com/@user/video/{video_id}"
    assert extract_video_id(www_url) == video_id


def test_post_id_extractors_ignore_query_suffixes_and_malformed_urls() -> None:
    video_url = "https://www.tiktok.com/@user/video/7123456789012345678?lang=en"
    photo_url = "https://www.tiktok.com/@user/photo/7123456789012345678/"
    assert extract_video_id(video_url) == "7123456789012345678"
    assert extract_post_id(photo_url) == "7123456789012345678"
    assert extract_video_id("https://www.tiktok.com/video/123") is None
    assert extract_post_id("https://[invalid/video/7123456789012345678") is None
    assert (
        extract_post_id(
            "https://example.com/@user/video/7123456789012345678"
        )
        is None
    )
    assert (
        extract_post_id(
            "https://www.tiktok.com/@user/video/12345678901234567890123456"
        )
        is None
    )


def test_output_paths_reject_traversal_and_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "downloads"
    validate_output_dir(output, create=True)
    assert safe_output_path(output, "7123456789012345678.mp4") == (
        output / "7123456789012345678.mp4"
    )
    with pytest.raises(ValueError):
        safe_output_path(output, "../outside.mp4")

    redirected = tmp_path / "redirected"
    redirected.mkdir()
    link = tmp_path / "link"
    link.symlink_to(redirected, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_output_dir(link)


def test_strict_artifact_validation_rejects_arbitrary_bytes(tmp_path: Path) -> None:
    invalid = tmp_path / "7123456789012345678.mp4"
    invalid.write_bytes(b"not media")
    valid = tmp_path / "7123456789012345679.mp4"
    valid.write_bytes(_VALID_MP4)

    assert not is_completed_artifact(invalid)
    assert is_completed_artifact(valid, media_type="video")
    assert get_downloaded_ids(tmp_path) == {"7123456789012345679"}


def test_slideshow_manifest_requires_exact_valid_files(tmp_path: Path) -> None:
    photo_dir = tmp_path / "7123456789012345678"
    photo_dir.mkdir()
    (photo_dir / "image_1.jpg").write_bytes(_VALID_JPEG)
    (photo_dir / ".tiktok-complete.json").write_text(
        '{"files": ["image_1.jpg"], "version": 1, '
        '"video_id": "7123456789012345678"}\n',
        encoding="utf-8",
    )

    assert is_completed_slideshow_artifact(photo_dir)
    (photo_dir / "image_2.jpg").write_bytes(_VALID_JPEG)
    assert not is_completed_slideshow_artifact(photo_dir)


def test_netscape_cookie_parser_preserves_http_only_and_expiry(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(
        "# Netscape HTTP Cookie File\n"
        "#HttpOnly_.tiktok.com\tTRUE\t/\tTRUE\t4102444800\tsid\tsecret\n"
        ".tiktok.com\tTRUE\t/account\tFALSE\t0\tplain\tvalue\n"
        "example.com\tTRUE\t/\tFALSE\t0\tother\tshould-not-load\n"
        ".tiktok.com\tTRUE\t/\tFALSE\t1\told\tignored\n",
        encoding="utf-8",
    )

    parsed = parse_netscape_cookies(cookies)
    assert parsed[0].http_only is True
    assert parsed[0].secure is True
    assert parsed[0].expires == 4102444800
    playwright = netscape_cookies_for_playwright(cookies)
    assert playwright[0]["httpOnly"] is True
    assert all(cookie["name"] != "old" for cookie in playwright)
    assert all(cookie["name"] != "other" for cookie in playwright)
    assert read_netscape_cookie_pairs(cookies, url="https://www.tiktok.com/") == {
        "sid": "secret",
    }
    assert len(netscape_cookie_jar(cookies)) == 3


def test_netscape_cookie_adapters_apply_domain_path_and_secure_rules(
    tmp_path: Path,
) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(
        ".tiktok.com\tTRUE\t/account\tTRUE\t0\taccount\tvalue\n"
        "www.tiktok.com\tFALSE\t/\tFALSE\t0\thost\tvalue\n",
        encoding="utf-8",
    )

    assert read_netscape_cookie_pairs(cookies, url="https://www.tiktok.com/account") == {
        "account": "value",
        "host": "value",
    }
    assert read_netscape_cookie_pairs(cookies, url="http://www.tiktok.com/account") == {
        "host": "value",
    }
    assert read_netscape_cookie_pairs(cookies, url="https://api.tiktok.com/account") == {
        "account": "value",
    }
    assert read_netscape_cookie_pairs(cookies, url="https://www.tiktok.com/") == {
        "host": "value",
    }


def test_isolated_cookie_file_does_not_share_path_or_mutate_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cookies.txt"
    source.write_text(".tiktok.com\tTRUE\t/\tFALSE\t0\tid\tvalue\n", encoding="utf-8")
    original = source.read_bytes()

    with isolated_cookie_file(source) as isolated:
        assert isolated is not None
        assert isolated != source
        assert isolated.read_bytes() == original
        isolated.write_bytes(b"isolated")

    assert source.read_bytes() == original
    assert not isolated.exists()


@pytest.mark.asyncio
async def test_request_limiter_serializes_request_starts() -> None:
    limiter = RequestLimiter(1.0)
    with (
        patch("tiktok_scraper.utils.time.monotonic", side_effect=[0.0, 0.0, 1.0]),
        patch("tiktok_scraper.utils.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        await limiter.acquire()
        await limiter.acquire()

    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_post_lock_releases_after_use(tmp_path: Path) -> None:
    from tiktok_scraper.utils import post_lock

    async with post_lock(tmp_path, "7123456789012345678", timeout=0.0):
        assert (tmp_path / ".7123456789012345678.lock").exists()
    if utils.fcntl is not None:
        # Unix uses a persistent inode with an advisory lock so a waiter cannot
        # race a lock-file unlink; the OS releases the lock if the owner crashes.
        assert (tmp_path / ".7123456789012345678.lock").exists()
    else:
        assert not (tmp_path / ".7123456789012345678.lock").exists()


def test_retry_after_parses_seconds_and_http_dates() -> None:
    assert parse_retry_after("3.5") == 3.5
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.UTC)
    assert parse_retry_after("Wed, 19 Aug 2026 12:00:05 GMT", now=now) == 5.0
    assert parse_retry_after("not-a-date") is None


def test_backoff_is_jittered_and_capped() -> None:
    with patch("tiktok_scraper.utils.random.uniform", return_value=1.0):
        assert bounded_backoff(4, base=1, cap=2) == 2


def test_backoff_does_not_violate_explicit_retry_after() -> None:
    with patch("tiktok_scraper.utils.random.uniform", return_value=0.0):
        assert bounded_backoff(1, base=1, cap=2, retry_after=120) == 2
