# TikTok Scraper

CLI tool to scrape all videos from a TikTok creator's profile.

## Features

- Download all videos from any public TikTok creator
- Photo slideshow support: downloads the actual images (not just the audio track)
- `--video-only` flag to skip audio-only posts with no video or photo content
- Automatic resume: skips already-downloaded videos and photo slideshows
- Concurrent downloads with configurable parallelism
- Best quality video+audio by default (separate streams are merged via ffmpeg)
- Layered discovery: yt-dlp, embed/channel recovery, and Playwright fallback
- Cookie support for authenticated sessions
- Chrome TLS impersonation, bounded retry backoff with jitter, and TikTok embed fallback for unreliable post pages
- Progress logging and detailed summary

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/download.html) (required when a post needs separate video/audio streams merged)
- [Playwright](https://playwright.dev/python/) Chromium browser (for fallback discovery)
- [curl-cffi](https://pypi.org/project/curl-cffi/) (used by yt-dlp impersonation for reliable TikTok extraction)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Tiktok-Scraper

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install with current yt-dlp TikTok support, curl-cffi, and dev tools
pip install -e ".[dev]"

# If you already installed earlier, upgrade yt-dlp to pick up latest TikTok fixes
pip install -U "yt-dlp>=2026.08.19"

# Install ffmpeg (required for merging video and audio streams)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
# Windows (using winget):
winget install ffmpeg
# Windows (using Chocolatey):
choco install ffmpeg

# Install the Playwright Chromium browser (needed for fallback only)
playwright install chromium
```

## Usage

### Basic usage

Download all videos from a creator:

```bash
tiktok-scraper <username>
```

Or run as a Python module:

```bash
python -m tiktok_scraper <username>
```

The `@` prefix is optional — both `tiktok` and `@tiktok` work.

Usernames may contain letters, numbers, periods, and underscores. `--limit`
accepts zero or a positive integer; `--limit 0` exits successfully without
contacting TikTok or creating the output directory. Concurrency is always at
least one.

### Options

```
Usage: tiktok-scraper [OPTIONS] USERNAME

  Scrape all videos from a TikTok creator's profile.

Options:
  -o, --output-dir DIRECTORY  Directory to save videos (default:
                              ./downloads/<username>/).
  --cookies FILE              Path to a Netscape-format cookies file
  -n, --limit INTEGER RANGE   Maximum number of videos to download.  [x>=0]
  --headless / --no-headless  Run browser in headless mode (default: headless)
  -c, --concurrency INTEGER RANGE
                              Number of concurrent downloads.  [default: 3;
                              x>=1]
  --video-only / --no-video-only
                               Skip audio-only posts (default: off)
  -v, --verbose               Enable debug logging
  --version                   Show the version and exit
  -h, --help                  Show this message and exit
```

### Examples

Download the 10 most recent videos from @tiktok:

```bash
tiktok-scraper tiktok --limit 10
```

Download to a custom directory with verbose logging:

```bash
tiktok-scraper charlidamelio -o ~/Videos/charli -v
```

Download with 5 concurrent downloads:

```bash
tiktok-scraper khaby.lame -c 5
```

Use cookies for authenticated access:

```bash
tiktok-scraper someuser --cookies cookies.txt
```

`--cookies` must point to a readable regular file. `--output-dir` may name an
existing directory or a new directory path whose parent directories can be
created.

Skip audio-only posts (download only videos and photo slideshows):

```bash
tiktok-scraper someuser --video-only
```

### Resume support

Running the same command again automatically skips videos and photo slideshows that have already been downloaded. Videos are detected by validated media files named `<id>.<ext>` and photo slideshows by validated directories named `<id>/`. New slideshows also contain a completion manifest recording the expected image set. This makes it safe to interrupt and restart the scraper.

Existing numeric-ID files and directories are never relocated. Video downloads
are written as `<id>.<ext>` and slideshow images are staged in
`.<id>.partial/` before being atomically renamed to `<id>/`. Partial staging
directories are not considered complete. When a post is retried, its stale
staging directory is removed while holding that post's lock. Staging directories
for other posts are left untouched.
yt-dlp resume files in the output root are retained and ignored for completion
detection, while completed files are preserved.

```bash
# First run: downloads 50 videos
tiktok-scraper tiktok --limit 50

# Second run: skips all 50, downloads nothing
tiktok-scraper tiktok --limit 50
```

## How it works

1. **Video discovery** uses yt-dlp's TikTok extractor as the primary method. It handles TikTok's anti-bot protections natively and is fast and reliable. If TikTok blocks profile hydration, the scraper fetches profile/video embed data to recover the account's secondary user ID, verifies the post author, then retries discovery through yt-dlp's `tiktokuser:<channel_id>` route. If that also fails, a Playwright-based browser fallback scrolls the profile page and collects validated video and photo URLs via DOM parsing, hydration data, and API response interception.

2. **Content classification** inspects each post's metadata (via yt-dlp's `extract_info`) before downloading. Posts are classified as one of three types:
   - **Video posts** -- have at least one video stream. Separate video and audio streams are merged into a single `.mp4` file via ffmpeg when available; a progressive video+audio format is selected when merging is unnecessary or ffmpeg is unavailable. This avoids incomplete output while retaining the best quality when possible.
   - **Photo slideshows** -- identified by a photo URL or semantic image-post data. The individual images are downloaded directly into a `<video_id>/` subdirectory and committed with a manifest only after every expected image validates.
   - **Audio-only posts** -- no video stream and no images (typically the audio track of a slideshow). Downloaded via yt-dlp by default, or skipped entirely when `--video-only` is active.

3. **Download execution** runs concurrently (default: 3 at a time) while a
     shared session limiter spaces request starts. The configured retry budget
     belongs to the complete post workflow, with bounded exponential backoff,
     jitter, and `Retry-After` support; permanent and local failures are not
     retried. Each post is claimed with a per-ID lock so concurrent processes do
     not publish the same artifact. Video downloads use yt-dlp with Chrome TLS
     impersonation. If TikTok's normal post page fails its anti-bot challenge,
     the scraper retries through TikTok's embed endpoint; this fallback
     prioritises completion and can be lower resolution than the primary
     download.

4. **Resume detection** scans the output directory for existing validated files
    named `<video_id>.<ext>` and directories named `<video_id>/` (photo
    slideshows), skipping any content that is already present. Temporary,
    corrupt, symlinked, and unrelated artifacts are never treated as completed.

### Rate limits and cookies

Keep concurrency modest (the default of 3 is intentional) and avoid repeated
large runs against the same account. yt-dlp retries are bounded and requests
are spaced, but TikTok can still rate-limit or challenge automated traffic.
Cookies are optional and should only be exported from an account you control.
Use a Netscape-format file, keep it private, never commit it, and do not share
it: cookies can grant access to the account until they expire or are revoked.
Cookie export files such as `cookies.txt` and `cookies.json` are ignored by
this repository's Git configuration.

The CLI summary reports video, audio, and photo-post downloads separately. A
non-zero failed count produces exit status 1, and inconclusive blocked
discovery also exits non-zero rather than being reported as an empty account.
Failure IDs and sanitized terminal reasons are included in the summary.

If a post requires separate video and audio streams with no progressive
fallback, ffmpeg is checked immediately before that post is downloaded. The
post is reported as failed rather than silently producing incomplete output
when ffmpeg is unavailable. Successful downloads require validated final
media artifacts; image transfers are bounded by the configured timeout and
size limit.

## Development

### Setup

```bash
pip install -e ".[dev]"
playwright install chromium
# Ensure ffmpeg is installed (see Installation above)
```

### Running tests

```bash
# All tests
pytest

# With verbose output
pytest -v

# With coverage
pytest --cov=tiktok_scraper
```

### Linting and formatting

```bash
# Check lint errors
ruff check .

# Auto-fix lint errors
ruff check . --fix

# Format code
ruff format .

# Type check
mypy tiktok_scraper/
```

## Project structure

```
LICENSE                 MIT license text
tiktok_scraper/
    __init__.py        Package init
    __main__.py        python -m support
    cli.py             Click CLI entry point
    config.py          ScraperConfig dataclass
    scraper.py         Video URL discovery (yt-dlp + Playwright fallback)
    downloader.py      yt-dlp download with concurrency
    models.py          VideoInfo, ScrapeResult dataclasses
    exceptions.py      Custom exception hierarchy
    utils.py           Helpers (video ID extraction, stealth browser, cookies)
tests/
    conftest.py        Shared fixtures
    test_cli.py        CLI tests
    test_entrypoint.py Module entry point test
    test_scraper.py    Scraper tests (yt-dlp + browser)
    test_downloader.py Downloader tests
```

## License

This project is licensed under the [MIT License](LICENSE).
