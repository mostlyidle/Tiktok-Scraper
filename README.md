# TikTok Scraper

CLI tool to scrape all videos from a TikTok creator's profile.

## Features

- Download all videos from any public TikTok creator
- Photo slideshow support: downloads the actual images (not just the audio track)
- `--video-only` flag to skip audio-only posts with no video or photo content
- Automatic resume: skips already-downloaded videos and photo slideshows
- Concurrent downloads with configurable parallelism
- Best quality video+audio by default (separate streams merged via ffmpeg)
- Two discovery strategies: yt-dlp (primary) with Playwright browser fallback
- Cookie support for authenticated sessions
- Retry logic with exponential backoff for both video and image downloads
- Progress logging and detailed summary

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/download.html) (required for merging video and audio streams)
- [Playwright](https://playwright.dev/python/) Chromium browser (for fallback discovery)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Tiktok-Scraper

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install the package with dependencies
pip install -e ".[dev]"

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

### Options

```
Usage: tiktok-scraper [OPTIONS] USERNAME

  Scrape all videos from a TikTok creator's profile.

Options:
  -o, --output-dir PATH       Directory to save videos
                               (default: ./downloads/<username>/)
  --cookies PATH              Path to a Netscape-format cookies file
  -n, --limit INTEGER         Maximum number of videos to download
  --headless / --no-headless  Run browser in headless mode (default: headless)
  -c, --concurrency INTEGER   Number of concurrent downloads (default: 3)
  --video-only / --no-video-only
                              Skip audio-only posts (default: off)
  -v, --verbose               Enable debug logging
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

Skip audio-only posts (download only videos and photo slideshows):

```bash
tiktok-scraper someuser --video-only
```

### Resume support

Running the same command again automatically skips videos and photo slideshows that have already been downloaded. Videos are detected by filename (e.g. `7123456789012345678.mp4`) and photo slideshows by directory name (e.g. `7123456789012345678/`). This makes it safe to interrupt and restart the scraper.

```bash
# First run: downloads 50 videos
tiktok-scraper tiktok --limit 50

# Second run: skips all 50, downloads nothing
tiktok-scraper tiktok --limit 50
```

## How it works

1. **Video discovery** uses yt-dlp's TikTok extractor as the primary method. It handles TikTok's anti-bot protections natively and is fast and reliable. If yt-dlp fails, a Playwright-based browser fallback scrolls the profile page and collects video and photo URLs via DOM parsing and API response interception.

2. **Content classification** inspects each post's metadata (via yt-dlp's `extract_info`) before downloading. Posts are classified as one of three types:
   - **Video posts** -- have at least one video stream. Video and audio streams are downloaded separately by yt-dlp and merged into a single `.mp4` file via ffmpeg. This ensures the best quality for both video and audio, since TikTok often serves them as separate streams.
   - **Photo slideshows** -- no video stream, but contain image URLs. The individual images are downloaded directly into a `<video_id>/` subdirectory.
   - **Audio-only posts** -- no video stream and no images (typically the audio track of a slideshow). Downloaded via yt-dlp by default, or skipped entirely when `--video-only` is active.

3. **Download execution** runs concurrently (default: 3 at a time) with automatic retry on failure. Video downloads are handled by yt-dlp with configurable retries. Image downloads use `urllib` with exponential backoff on transient HTTP errors (429, 5xx) and network failures.

4. **Resume detection** scans the output directory for existing files named `<video_id>.<ext>` and directories named `<video_id>/` (photo slideshows), skipping any content that is already present.

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
    test_scraper.py    Scraper tests (yt-dlp + browser)
    test_downloader.py Downloader tests
```
