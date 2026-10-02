# Photo Organizer Studio

An autonomous, non-destructive photo organization and curation suite with a local interactive Web UI, command-line interface, multi-hash deduplicator, text-forward quarantine, adaptive quality enhancer, offline AI semantic search, face clustering, GPS world mapping, burst shot selection, and memory flashbacks.

* **[Architecture & Design Specification](DESIGN.md)**: In-depth technical specifications, mathematical scoring formulations, and subsystem decomposition.
* **[Operations & How-To Runbook](RUNBOOK.md)**: Practical workflows, Web UI guide, SOPs, performance tuning, and troubleshooting.

---

## Key Features

1. **Interactive Web UI Studio**:
   - Modern, responsive local web dashboard built with FastAPI, Tailwind CSS, and Leaflet.js.
   - 9 dedicated workspaces:
     - **Options & Controls**: Live status cards, configuration controls, and one-click EXIF date back-filling.
     - **Duplicates**: Side-by-side visual duplicate comparison with split-screen inspection, resolution badges, and one-click keeper switching.
     - **Triage Queue**: Fast review queue for borderline WhatsApp forwards, memes, and receipts.
     - **AI Search**: 100% offline semantic search querying natural language tags, seasons, times of day, colors, and locations (e.g. `sunset beach`, `winter golden hour`, `red car`).
     - **World Map**: Interactive Leaflet map plotting geotagged photos with reverse-geocoded city/country clusters.
     - **Burst Shots**: Identifies rapid-fire sequences ($\le 3$s), calculates composite sharpness & exposure quality scores, and automatically selects the "Best Shot".
     - **People Albums**: Local offline face detection and spatial color clustering to group photos by person, with inline rename support.
     - **Flashbacks**: "On This Day" anniversary memory cards celebrating photos taken on today's calendar date in previous years.
     - **History & Rollback**: Complete SQLite transaction history with instant visual rollback.

2. **100% Offline & Zero Cloud Dependency**:
   - Every single feature runs strictly on your local machine.
   - Zero cloud uploads, zero API keys required, zero telemetry, and complete privacy.

3. **Format Upgrades & Companion Pairing**:
   - **Native Apple HEIC/HEIF**: Direct decoding via `pillow-heif`.
   - **Camera RAW Previews**: Embedded high-resolution preview extraction for Canon (`.cr2`, `.cr3`), Nikon (`.nef`), Sony (`.arw`), Adobe (`.dng`), Olympus (`.orf`), Panasonic (`.rw2`), and Pentax (`.pef`).
   - **Live Photos & Sidecars**: iPhone Live Photos (`IMG_0001.HEIC` + `IMG_0001.MOV`), iOS edit sidecars (`.AAE`), and Adobe Lightroom/Darktable metadata (`.XMP`) are paired by file stem and moved atomically. Built-in Web UI modal plays Live Photo companion videos.

4. **Standalone Video Hierarchy**:
   - Standalone video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`, `.3gp`, etc.) are organized under a dedicated `video/` directory following the same `video/YYYY/YYYY-MM/` temporal convention, keeping still images and videos cleanly separated.

5. **Multi-Hash Perceptual Deduplication**:
   - Difference hashing (dHash) combined with 2D Discrete Cosine Transform (pHash) for resilience against scaling, minor crops, and slight color variations.
   - 8-bit band indexing (Locality-Sensitive Hashing) and aspect-ratio bucketing avoid expensive $O(N^2)$ comparisons on large libraries.

6. **Adaptive In-Place Quality Enhancement**:
   - Enhances underexposed, muted, or soft photos using adaptive CLAHE contrast stretching and unsharp masking.
   - Strictly enforces maximum file size increase limits (default 20%).
   - Restores original access and modification timestamps (`atime`/`mtime`) and maintains OS creation dates.

7. **EXIF Metadata Back-Filling**:
   - Fixes photos stripped of metadata (e.g. WhatsApp images, downloaded memes, scanned prints).
   - Writes verified timestamps back into standard EXIF headers (`DateTimeOriginal`, `DateTimeDigitized`, `DateTime`) so operating systems, Google Photos, and Apple Photos sort them accurately.

8. **One-Click Reversibility & Rollback (`--undo`)**:
   - Atomic SQLite transaction ledger (`.photo_organizer_journal.db`) recording every file move, enhancement, and directory removal.
   - Run `python photo_organizer.py --undo` or click the rollback button in the Web UI at any time to restore files back to their exact original locations.

9. **Persistent Speed Cache**:
   - Local SQLite cache (`.photo_cache.sqlite`) indexing file sizes and modification times for near-instant re-scans.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Web UI Studio

**Windows (One-Click)**:
Double-click `launch_ui.bat` or drag and drop any picture folder directly onto `launch_ui.bat`.

**Command Line**:
```bash
python photo_organizer.py --ui /path/to/pictures
```

Open `http://127.0.0.1:8080` in your web browser.

Customize host and port:
```bash
python photo_organizer.py --ui --host 0.0.0.0 --port 9000 /path/to/pictures
```

---

### 2. Command-Line Interface (CLI)

Preview actions without touching any files:
```bash
python photo_organizer.py --dry-run /path/to/pictures
```

Organize, deduplicate, and enhance in-place:
```bash
python photo_organizer.py /path/to/pictures
```

Rollback the last organization run:
```bash
python photo_organizer.py --undo /path/to/pictures
```

---

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `folder` | Target folder containing photos and videos (supports `%PERSONALHOME%` or relative paths). | Current Directory |
| `--ui`, `--serve` | Launches the interactive local Web UI Studio. | False |
| `--undo` | Rolls back the last completed organization run. | False |
| `--dry-run` | Previews all actions without moving or modifying files. | False |
| `--skip-enhance` | Skips image enhancement; only organizes and deduplicates. | False |
| `--skip-non-relevant`| Skips screening text-forward/text-heavy images. | False |
| `--visual-threshold` | Perceptual sensitivity (0-64). Lower is stricter. | `5` |
| `--max-size-increase`| Max percentage file-size increase permitted during enhancement. | `20.0` |
| `--enhancement-mode` | Strategy: `fast`, `balanced`, or `best`. | `balanced` |
| `--workers` | Number of worker threads for scanning and processing. | CPU count (max 8) |
| `--no-cache` | Disables the persistent metadata and hash cache. | False |

---

## Directory Organization Structure

After organizing, files are structured cleanly into:

```text
Pictures/
  ├── 2024/
  │   └── 2024-01/
  │       ├── IMG_0001.JPG
  │       ├── IMG_0001.MOV         <-- Paired Live Photo companion
  │       └── IMG_0001.XMP         <-- Paired metadata sidecar
  ├── video/
  │   └── 2024/
  │       └── 2024-01/
  │           └── VID_20240101.mp4 <-- Standalone video collection
  ├── to-be-deleted/
  │   └── 2024/
  │       └── 2024-01/
  │           └── IMG_0001_copy.JPG
  ├── non-relevant/
  │   └── 2024/
  │       └── 2024-01/
  │           └── whatsapp_forward.jpg
  └── maybe-non-relevant/
      └── 2024/
          └── 2024-01/
              └── receipt_or_invitation.jpg
```

---

## Running Tests

Execute the full unit test suite:

```bash
python -m unittest discover tests -v
```

