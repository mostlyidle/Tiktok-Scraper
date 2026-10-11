"""Helpers: video ID extraction, stealth browser setup, rate limiting."""

import asyncio
import contextlib
import datetime as dt
import email.utils
import http.cookiejar
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERN = re.compile(
    r"^/@[A-Za-z0-9._]{1,24}/video/(\d{15,25})/?\Z",
    re.IGNORECASE,
)
_POST_ID_PATTERN = re.compile(
    r"^/@[A-Za-z0-9._]{1,24}/(?:video|photo)/(\d{15,25})/?\Z",
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
_TEMPORARY_SUFFIXES = (".part", ".ytdl", ".tmp")
_VALID_VIDEO_ID_PATTERN = re.compile(r"\d{15,25}\Z")
_STALE_LOCK_GRACE_SECONDS = 30.0
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_IMAGE_SUFFIXES = (".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp")
_VIDEO_SUFFIXES = (".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm")
_AUDIO_SUFFIXES = (".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm")
_SLIDESHOW_MANIFEST_NAME = ".tiktok-complete.json"
_URL_QUERY_LOG_PATTERN = re.compile(r"(?i)(https?://[^\s<>\"']+)\?[^\s<>\"']*")
_QUERY_PARAM_LOG_PATTERN = re.compile(r"([?&][A-Za-z0-9_.-]+)=([^&\s]+)")
_SENSITIVE_LOG_PATTERN = re.compile(
    r"(?i)\b(?:cookie(?:file)?|cookies?|password|secret|token|auth(?:entication)?)"
    r"\s*[:=]\s*[^\s,;]+"
)


def redact_log_value(value: object) -> str:
    """Remove URLs, query strings, and credential-like values from logs."""
    message = str(value)
    message = _URL_QUERY_LOG_PATTERN.sub(r"\1?[query redacted]", message)
    message = _QUERY_PARAM_LOG_PATTERN.sub(r"\1=[redacted]", message)
    return _SENSITIVE_LOG_PATTERN.sub(
        lambda match: f"{match.group(0).split('=', 1)[0].split(':', 1)[0]}=[redacted]",
        message,
    )


@dataclass(frozen=True, slots=True)
class NetscapeCookie:
    """One cookie from a Netscape/Mozilla cookie-jar file.

    The Netscape format stores a domain-match flag separately from the domain
    itself.  Keeping both values avoids accidentally broadening a host-only
    cookie when adapting it for another HTTP client.
    """

    domain: str
    include_subdomains: bool
    path: str
    secure: bool
    expires: int | None
    name: str
    value: str
    http_only: bool = False

    def is_expired(self, *, now: dt.datetime | None = None) -> bool:
        """Return whether this cookie has a positive expiry in the past."""
        if self.expires is None:
            return False
        current = now or dt.datetime.now(dt.UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.UTC)
        return self.expires <= int(current.timestamp())

    def for_playwright(
        self, *, now: dt.datetime | None = None
    ) -> dict[str, object] | None:
        """Adapt this cookie to Playwright's cookie dictionary format."""
        if self.is_expired(now=now):
            return None
        domain = self.domain
        if self.include_subdomains and not domain.startswith("."):
            domain = f".{domain}"
        cookie: dict[str, object] = {
            "name": self.name,
            "value": self.value,
            "domain": domain,
            "path": self.path,
            "secure": self.secure,
            "httpOnly": self.http_only,
        }
        if self.expires is not None:
            cookie["expires"] = self.expires
        return cookie

    def for_cookiejar(self) -> http.cookiejar.Cookie | None:
        """Adapt this cookie to Python's standards-aware CookieJar."""
        if self.is_expired():
            return None
        domain = self.domain
        if self.include_subdomains and not domain.startswith("."):
            domain = f".{domain}"
        rest: dict[str, str] = {"HttpOnly": ""} if self.http_only else {}
        return http.cookiejar.Cookie(
            version=0,
            name=self.name,
            value=self.value,
            port=None,
            port_specified=False,
            domain=domain,
            domain_specified=True,
            domain_initial_dot=domain.startswith("."),
            path=self.path,
            path_specified=True,
            secure=self.secure,
            expires=self.expires,
            discard=self.expires is None,
            comment=None,
            comment_url=None,
            rest=rest,
            rfc2109=False,
        )


def parse_netscape_cookies(
    cookies_file: Path | None,
) -> list[NetscapeCookie]:
    """Parse a Netscape cookie file without mutating it.

    Blank lines and ordinary comments are ignored.  ``#HttpOnly_`` is a
    cookie-domain prefix, not a comment.  Malformed rows are ignored rather
    than converted into permissive session cookies.  A zero or negative
    expiry is the format's session-cookie sentinel; positive timestamps are
    retained so adapters can enforce expiry semantics.
    """
    if cookies_file is None:
        return []
    path = Path(cookies_file)
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"cookies_file must be a regular file: {path}")

    parsed: list[NetscapeCookie] = []
    with path.open(encoding="utf-8", errors="replace") as cookie_stream:
        for raw_line in cookie_stream:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            fields = line.split("\t", 6)
            if len(fields) != 7:
                continue
            raw_domain, include_subdomains, cookie_path, secure, expiry, name, value = (
                fields
            )

            http_only = raw_domain.startswith("#HttpOnly_")
            if raw_domain.startswith("#") and not http_only:
                continue
            domain = raw_domain.removeprefix("#HttpOnly_").strip().lower()
            if not domain or any(char.isspace() for char in domain):
                continue
            if not cookie_path.startswith("/"):
                continue
            if not name or any(char.isspace() for char in name):
                continue
            if include_subdomains.upper() not in {"TRUE", "FALSE"}:
                continue
            if secure.upper() not in {"TRUE", "FALSE"}:
                continue
            try:
                expiry_value = int(expiry)
            except ValueError:
                continue

            parsed.append(
                NetscapeCookie(
                    domain=domain,
                    include_subdomains=include_subdomains.upper() == "TRUE",
                    path=cookie_path,
                    secure=secure.upper() == "TRUE",
                    expires=expiry_value if expiry_value > 0 else None,
                    name=name,
                    value=value,
                    http_only=http_only,
                )
            )
    return parsed


def netscape_cookies_for_playwright(
    cookies_file: Path | None,
) -> list[dict[str, object]]:
    """Return non-expired Netscape cookies in Playwright's input format."""
    cookies: list[dict[str, object]] = []
    for cookie in parse_netscape_cookies(cookies_file):
        if not _is_trusted_tiktok_cookie_domain(cookie.domain):
            continue
        playwright_cookie = cookie.for_playwright()
        if playwright_cookie is not None:
            cookies.append(playwright_cookie)
    return cookies


def _is_trusted_tiktok_cookie_domain(domain: str) -> bool:
    """Return whether a cookie belongs to TikTok's controlled domains."""
    normalized = domain.lstrip(".").casefold()
    return normalized in _TIKTOK_HOSTS or normalized.endswith(".tiktok.com")


def netscape_cookie_jar(cookies_file: Path | None) -> http.cookiejar.CookieJar:
    """Return a standards-aware cookie jar for curl-compatible clients."""
    jar = http.cookiejar.CookieJar()
    for cookie in parse_netscape_cookies(cookies_file):
        jar_cookie = cookie.for_cookiejar()
        if jar_cookie is not None:
            jar.set_cookie(jar_cookie)
    return jar


def read_netscape_cookie_pairs(
    cookies_file: Path | None,
    *,
    url: str | None = None,
) -> dict[str, str]:
    """Return applicable cookie name/value pairs for simple HTTP clients.

    This adapter is intentionally lossy because a plain dictionary cannot
    represent duplicate names across domains or paths.  When *url* is given,
    domain, path, secure, and expiry rules are applied before choosing the
    last matching value.
    """
    try:
        parsed_url = urlsplit(url) if url else None
        host = parsed_url.hostname.lower() if parsed_url and parsed_url.hostname else None
    except ValueError:
        return {}
    request_path = parsed_url.path or "/" if parsed_url else "/"
    secure_request = parsed_url is not None and parsed_url.scheme.lower() == "https"
    pairs: dict[str, str] = {}
    now = dt.datetime.now(dt.UTC)

    for cookie in parse_netscape_cookies(cookies_file):
        if cookie.is_expired(now=now):
            continue
        if parsed_url is not None:
            domain = cookie.domain.lstrip(".")
            if host is None or (
                host != domain
                and not (cookie.include_subdomains and host.endswith(f".{domain}"))
            ):
                continue
            if not _cookie_path_matches(cookie.path, request_path):
                continue
            if cookie.secure and not secure_request:
                continue
        pairs[cookie.name] = cookie.value
    return pairs


def _cookie_path_matches(cookie_path: str, request_path: str) -> bool:
    """Apply RFC-style path matching for the dictionary adapter."""
    if request_path == cookie_path:
        return True
    if not request_path.startswith(cookie_path):
        return False
    return cookie_path.endswith("/") or request_path[len(cookie_path) :].startswith("/")


@contextlib.contextmanager
def isolated_cookie_file(cookies_file: Path | None) -> Iterator[Path | None]:
    """Yield an isolated cookie-file copy for a single client invocation.

    Some HTTP clients can persist cookie changes when closing.  A separate
    per-invocation copy prevents concurrent downloader workers from corrupting
    the configured source file or each other's state.
    """
    if cookies_file is None or not Path(cookies_file).exists():
        yield cookies_file
        return

    source = Path(cookies_file)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"cookies_file must be a regular file: {source}")
    contents = source.read_bytes()
    with tempfile.TemporaryDirectory(prefix="tiktok-cookies-") as temp_dir:
        isolated = Path(temp_dir) / "cookies.txt"
        isolated.write_bytes(contents)
        isolated.chmod(0o600)
        yield isolated


class RequestLimiter:
    """Serialize request starts with one minimum interval per session."""

    def __init__(self, min_interval: float = 0.0) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        self._min_interval = float(min_interval)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        """Wait until this session may start another network operation."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval

    async def wait(self) -> None:
        """Compatibility alias for :meth:`acquire`."""
        await self.acquire()


def _lock_is_active(lock_path: Path) -> bool:
    """Return whether a lock file is currently held by another process."""
    if not lock_path.exists():
        return False
    if fcntl is None:
        try:
            owner_pid = int(lock_path.read_text(encoding="ascii").splitlines()[0])
        except (OSError, ValueError, IndexError):
            try:
                return (
                    time.time() - lock_path.stat().st_mtime
                    <= _STALE_LOCK_GRACE_SECONDS
                )
            except OSError:
                return True
        if owner_pid == os.getpid():
            return True
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True
    try:
        descriptor = os.open(lock_path, os.O_RDWR | _NOFOLLOW)
    except OSError:
        return True
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


@contextlib.asynccontextmanager
async def post_lock(
    output_dir: Path,
    post_id: str,
    *,
    timeout: float = 300.0,
) -> AsyncIterator[None]:
    """Claim one post directory across workers and scraper processes."""
    if not is_valid_video_id(post_id):
        raise ValueError("post_id must contain 15 to 25 decimal digits")
    if timeout < 0:
        raise ValueError("post lock timeout must be non-negative")

    root = validate_output_dir(output_dir, create=True)
    lock_path = safe_output_path(root, f".{post_id}.lock")
    deadline = time.monotonic() + timeout

    if fcntl is not None:
        file_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | _NOFOLLOW,
            0o600,
        )
        try:
            while True:
                try:
                    fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for post lock {post_id}"
                        ) from None
                    await asyncio.sleep(
                        min(0.2, max(0.0, deadline - time.monotonic()))
                    )
            os.ftruncate(file_descriptor, 0)
            os.write(file_descriptor, f"{os.getpid()}\n".encode())
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(file_descriptor)
            raise
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(file_descriptor)
        return

    marker_descriptor: int | None = None
    while marker_descriptor is None:
        try:
            marker_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(marker_descriptor, f"{os.getpid()}\n".encode())
        except FileExistsError:
            if fcntl is None and not _lock_is_active(lock_path):
                with contextlib.suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for post lock {post_id}"
                ) from None
            await asyncio.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        except BaseException:
            if marker_descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(marker_descriptor)
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                marker_descriptor = None
            raise
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(marker_descriptor)
        with contextlib.suppress(OSError):
            lock_path.unlink()


def extract_video_id(url: str) -> str | None:
    """Extract the numeric video ID from a TikTok video URL.

    Args:
        url: A TikTok video URL like
            ``https://www.tiktok.com/@user/video/7123456789012345678``.

    Returns:
        The video ID string, or ``None`` if the URL does not match.
    """
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.netloc and not _is_tiktok_url(parsed):
            return None
        path = parsed.path
    except ValueError:
        return None
    match = _VIDEO_ID_PATTERN.search(path)
    return match.group(1) if match else None


def extract_post_id(url: str) -> str | None:
    """Extract the numeric post ID from a TikTok video or photo URL.

    Recognises both ``/video/<id>`` and ``/photo/<id>`` URL patterns.

    Args:
        url: A TikTok post URL like
            ``https://www.tiktok.com/@user/video/7123456789012345678``
            or ``https://www.tiktok.com/@user/photo/7123456789012345678``.

    Returns:
        The post ID string, or ``None`` if the URL does not match.
    """
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.netloc and not _is_tiktok_url(parsed):
            return None
        path = parsed.path
    except ValueError:
        return None
    match = _POST_ID_PATTERN.search(path)
    return match.group(1) if match else None


def _is_tiktok_url(parsed: SplitResult) -> bool:
    """Return whether a parsed absolute URL is an official TikTok URL."""
    hostname = parsed.hostname
    scheme = parsed.scheme
    return (
        isinstance(hostname, str)
        and hostname.casefold() in _TIKTOK_HOSTS
        and scheme.casefold() == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
    )


def is_valid_video_id(video_id: str) -> bool:
    """Return whether *video_id* is safe as a downloader path component."""
    return isinstance(video_id, str) and _VALID_VIDEO_ID_PATTERN.fullmatch(video_id) is not None


def validate_output_dir(output_dir: Path, *, create: bool = False) -> Path:
    """Validate an output directory and return its resolved path.

    The output directory itself may not be a symlink. Parent symlinks such as
    macOS's ``/tmp -> /private/tmp`` are resolved and retained as a stable root.
    """
    raw_path = Path(output_dir)
    if any(part == ".." for part in raw_path.parts):
        raise ValueError("output_dir must not contain parent traversal")

    absolute = Path(os.path.abspath(raw_path))
    if absolute.is_symlink():
        raise ValueError(f"output_dir must not be a symlink: {output_dir}")
    created = False
    if create and not absolute.exists():
        try:
            absolute.mkdir(parents=True, exist_ok=True)
            created = True
        except OSError as exc:
            raise ValueError(f"could not create output_dir: {absolute}") from exc
        if absolute.is_symlink():
            raise ValueError(f"output_dir must not be a symlink: {output_dir}")
        if created:
            try:
                absolute.chmod(0o700)
            except OSError as exc:
                raise ValueError(f"could not secure output_dir: {absolute}") from exc

    if absolute.exists() and not absolute.is_dir():
        raise ValueError(f"output_dir must be a regular directory: {output_dir}")
    try:
        resolved = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"could not resolve output_dir: {output_dir}") from exc
    return resolved


def safe_output_path(output_dir: Path, *parts: str | Path) -> Path:
    """Return a contained, non-symlink child path under *output_dir*."""
    root = validate_output_dir(output_dir)
    clean_parts: list[str] = []
    for part in parts:
        child = Path(part)
        if child.is_absolute() or any(piece in {"", ".", ".."} for piece in child.parts):
            raise ValueError("output artifact path must be a relative child path")
        clean_parts.extend(child.parts)
    if not clean_parts:
        raise ValueError("output artifact path must name a child")
    candidate = root.joinpath(*clean_parts)
    _reject_symlink_components(candidate, root=root)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"could not resolve output artifact path: {candidate}") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("output artifact path escapes output_dir")
    return candidate


def _reject_symlink_components(path: Path, *, root: Path | None = None) -> None:
    """Reject symlinks below a trusted root without following them."""
    absolute = Path(os.path.abspath(path))
    trusted_root = (
        Path(os.path.abspath(root)) if root is not None else Path(absolute.anchor)
    )
    current = trusted_root
    relative_parts = absolute.relative_to(trusted_root).parts
    for component in relative_parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"symlink path component is not allowed: {current}")


def get_downloaded_ids(output_dir: Path) -> set[str]:
    """Scan *output_dir* and return the set of already-downloaded video IDs.

    Video files are expected to be named ``<video_id>.<ext>``.
    Photo slideshows are stored as directories named ``<video_id>/``.
    """
    root = validate_output_dir(output_dir)
    if not root.exists():
        return set()
    ids: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink():
            continue
        if path.is_file() and is_valid_video_id(path.stem) and is_completed_artifact(path):
            ids.add(path.stem)
        elif (
            path.is_dir()
            and is_valid_video_id(path.name)
            and is_completed_slideshow_artifact(path)
        ):
            ids.add(path.name)
    return ids


def is_completed_artifact(path: Path, *, media_type: str | None = None) -> bool:
    """Return whether *path* is a strict, nonempty media artifact.

    ``media_type`` may be ``"video"``, ``"audio"``, or ``"image"``.  When
    omitted, the suffix still has to be a known media suffix and its small
    file signature must be plausible; arbitrary nonempty files are rejected.
    """
    path = Path(path)
    if path.is_symlink() or path.name.lower().endswith(_TEMPORARY_SUFFIXES):
        return False
    if path.is_file():
        if media_type not in {None, "video", "audio", "image"}:
            return False
        try:
            if path.stat().st_size <= 0:
                return False
            with path.open("rb") as media_file:
                header = media_file.read(32)
        except OSError:
            return False
        return _has_media_signature(path.suffix.lower(), header, media_type)
    if path.is_dir() and media_type in {None, "slideshow"}:
        return is_completed_slideshow_artifact(path)
    return False


def is_completed_slideshow_artifact(path: Path) -> bool:
    """Return whether a slideshow directory contains only valid images."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    manifest_path = path / _SLIDESHOW_MANIFEST_NAME
    manifest: dict[str, object] | None = None
    if manifest_path.is_symlink():
        return False
    if manifest_path in children:
        try:
            with manifest_path.open(encoding="utf-8") as manifest_stream:
                loaded = json.load(manifest_stream)
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(loaded, dict):
            return False
        manifest = loaded
        children.remove(manifest_path)

    image_names = {
        child.name
        for child in children
        if not child.is_symlink() and child.is_file()
    }
    if not image_names or len(image_names) != len(children):
        return False
    if manifest is not None:
        if manifest.get("video_id") != path.name:
            return False
        expected = manifest.get("files")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(name, str) for name in expected)
            or len(set(expected)) != len(expected)
            or set(expected) != image_names
        ):
            return False
    return all(
        is_completed_artifact(path / name, media_type="image")
        for name in image_names
    )


def _has_media_signature(suffix: str, header: bytes, media_type: str | None) -> bool:
    """Perform conservative container/signature validation without ffprobe."""
    if media_type == "image" and suffix not in _IMAGE_SUFFIXES:
        return False
    if media_type == "video" and suffix not in _VIDEO_SUFFIXES:
        return False
    if media_type == "audio" and suffix not in _AUDIO_SUFFIXES:
        return False
    if media_type is None and suffix not in (
        *_IMAGE_SUFFIXES,
        *_VIDEO_SUFFIXES,
        *_AUDIO_SUFFIXES,
    ):
        return False

    image_signature = (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
        or (len(header) >= 12 and header[4:8] == b"ftyp" and b"avif" in header[8:])
    )
    if media_type == "image":
        return image_signature
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    if suffix == ".avif":
        return len(header) >= 12 and header[4:8] == b"ftyp" and b"avif" in header[8:]
    if suffix in {".mp4", ".m4v", ".m4a", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix in {".mkv", ".webm"}:
        return header.startswith(b"\x1a\x45\xdf\xa3")
    if suffix == ".avi":
        return header.startswith(b"RIFF") and header[8:12] == b"AVI "
    if suffix == ".mpeg":
        return header.startswith(b"\x47") or header.startswith(b"\x00\x00\x01\xba")
    if suffix == ".mp3":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        )
    if suffix in {".aac", ".opus", ".ogg"}:
        return header.startswith(b"OggS") if suffix != ".aac" else (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0
        )
    if suffix == ".wav":
        return header.startswith(b"RIFF") and header[8:12] == b"WAVE"
    if suffix == ".flac":
        return header.startswith(b"fLaC")
    return False


def parse_retry_after(
    value: str | None, *, now: dt.datetime | None = None
) -> float | None:
    """Parse a Retry-After seconds value or HTTP date."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        current = now or dt.datetime.now(dt.UTC)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=dt.UTC)
        seconds = (retry_at - current).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, min(seconds, 24 * 60 * 60))


def bounded_backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
    retry_after: float | None = None,
) -> float:
    """Return capped exponential backoff with bounded positive jitter."""
    exponential = min(cap, base * (2 ** max(0, attempt - 1)))
    retry_floor = (
        None
        if retry_after is None
        else min(cap, max(0.0, retry_after))
    )
    floor = exponential if retry_floor is None else max(exponential, retry_floor)
    jitter = float(random.uniform(0.0, min(1.0, max(0.0, floor * 0.25))))
    return float(min(cap, floor + jitter))


async def random_delay(min_seconds: float, max_seconds: float) -> None:
    """Sleep for a random duration between *min_seconds* and *max_seconds*."""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)


async def create_stealth_browser_context(
    headless: bool = True,
    cookies_file: Path | None = None,
) -> tuple[Playwright, Browser, BrowserContext]:
    """Create a Playwright browser context with stealth-like settings.

    Returns a ``(playwright_instance, browser, browser_context)`` tuple.
    The caller is responsible for closing all three when done::

        pw, browser, context = await create_stealth_browser_context()
        try:
            ...
        finally:
            await context.close()
            await browser.close()
            await pw.stop()

    Args:
        headless: Whether to run the browser in headless mode.
        cookies_file: Optional path to a Netscape-format cookies file.

    Returns:
        Tuple of (playwright instance, browser, browser context).
    """
    pw = await async_playwright().start()
    browser: Browser | None = None
    context: BrowserContext | None = None
    try:
        # Keep Chromium's sandbox enabled. The browser consumes remote content
        # and may receive authenticated cookies, so disabling it is unsafe.
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            java_script_enabled=True,
        )

        # Remove the webdriver navigator property that flags automation.
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

        if cookies_file is not None:
            await _load_cookies(context, cookies_file)

        return pw, browser, context
    except BaseException:
        if context is not None:
            with contextlib.suppress(BaseException):
                await context.close()
        if browser is not None:
            with contextlib.suppress(BaseException):
                await browser.close()
        with contextlib.suppress(BaseException):
            await pw.stop()
        raise


async def _load_cookies(context: BrowserContext, cookies_file: Path) -> None:
    """Load Netscape-format cookies into the browser context."""
    if not cookies_file.exists():
        logger.warning("Cookies file %s does not exist; skipping.", cookies_file)
        return

    cookies = netscape_cookies_for_playwright(cookies_file)

    if cookies:
        await context.add_cookies(cookies)  # type: ignore[arg-type]
        logger.info("Loaded %d cookies from %s.", len(cookies), cookies_file)


def check_ffmpeg() -> bool:
    """Check whether ffmpeg is available on the system PATH.

    Logs a warning if ffmpeg is not found, since it is required for
    merging separate video and audio streams downloaded by yt-dlp.

    Returns:
        ``True`` if ffmpeg is found, ``False`` otherwise.
    """
    if shutil.which("ffmpeg") is not None:
        return True

    logger.warning(
        "ffmpeg not found on PATH. ffmpeg is required to merge video and "
        "audio streams. Downloads may lack audio. "
        "Install ffmpeg: https://ffmpeg.org/download.html",
    )
    return False
