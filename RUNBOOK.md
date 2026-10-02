# Photo Organizer Studio — Operations & How-To Runbook

This runbook provides step-by-step operational instructions for managing, organizing, deduplicating, and curating photo and video libraries using **Photo Organizer Studio**.

---

## 1. Quickstart & Installation

### 1.1 Prerequisites
- **Operating System**: Windows 10/11, macOS (12+), or Linux (Ubuntu 20.04+, Debian 11+, Fedora, etc.).
- **Python**: Version 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit recommended).

### 1.2 Setup Environment
Open your terminal or command prompt in the project root directory:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 1.3 Verify Installation
Execute the unit test suite to confirm all vision, deduplication, and format libraries are operating correctly:

```bash
python -m unittest discover tests -v
```
You should see all 35 tests pass with `OK`.

---

## 2. Execution Modes

### Mode A: Web UI Studio (Recommended)

The Web UI Studio offers an interactive dashboard with side-by-side duplicate comparison, world mapping, face clustering, burst selection, and real-time progress indicators.

#### Windows (One-Click Launch)
- **Method 1**: Double-click `launch_ui.bat`. Your default web browser will open `http://127.0.0.1:8080` automatically.
- **Method 2 (Drag & Drop)**: Drag any photo folder directly onto the `launch_ui.bat` icon. The Studio will launch with that folder pre-loaded.

#### Command-Line Launch
```bash
# Launch with target directory pre-selected:
python photo_organizer.py --ui /path/to/photos

# Launch with custom port or network binding:
python photo_organizer.py --ui --host 127.0.0.1 --port 9000 /path/to/photos
```

---

### Mode B: Command-Line Interface (Headless CLI)

For headless servers, automation scripts, or terminal power users:

#### 1. Preview Run (Dry-Run — No files modified)
Always recommended before running on a new library:
```bash
python photo_organizer.py --dry-run /path/to/photos
```

#### 2. Standard Organization Run
Moves photos into `YYYY/YYYY-MM/`, standalone videos into `video/YYYY/YYYY-MM/`, duplicates into `to-be-deleted/`, and text-heavy graphics into `non-relevant/`:
```bash
python photo_organizer.py /path/to/photos
```

#### 3. Organization with Quality Enhancement
Analyzes and enhances underexposed, low-contrast photos in-place while enforcing a maximum 20% file size increase limit:
```bash
python photo_organizer.py --enhancement-mode balanced /path/to/photos
```

#### 4. Instant Rollback (`--undo`)
Accidentally moved something or want to revert back to the exact initial state?
```bash
python photo_organizer.py --undo /path/to/photos
```
The transaction journal will restore all files and directory structures back to their pre-run states.

---

## 3. Web UI Studio Walkthrough

When you open `http://127.0.0.1:8080`, the interface cleanly separates **Runtime Organization Actions** from **Library Query & Viewing**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Photo Organizer Studio                                                │
│  Target: [/path/to/photos            ] [ Set Folder ] [ Scan Library ] │
├────────────────────────────────────────────────────────────────────────┤
│  [KPI: Total] [Duplicates] [Triage] [Map Pins] [Bursts] [People]       │
├────────────────────────────────────────────────────────────────────────┤
│  [ ⚡ Curate & Organize (Actions) ]   [ 🧭 Explore & View (Library) ]  │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────┐ │
│  │ ⚙️ Options & EXIF               │   │ 👥 People Albums (SFace)    │ │
│  │ 🖼️ Duplicates (Interactive)     │   │ ✨ Flashbacks (Memories)    │ │
│  │ 📑 Triage Queue (Memes/Junk)    │   │ 🗺️ World Map (GPS Clusters) │ │
│  │ ⚡ Burst Shots (Best Selector)  │   │ 🔍 AI Semantic Search       │ │
│  └─────────────────────────────────┘   │ 📜 History & Rollback (Undo)│ │
│                                        └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Initial Scan & Folder Selection
1. Enter the path to your photo library in the top bar (e.g. `%PERSONALHOME%\Pictures` or `/path/to/photos`).
2. Click **Set Folder**.
   - If the folder was previously scanned or organized, the Studio **immediately re-hydrates** your People albums, Flashbacks, World Map, and History from the persistent cache without requiring a re-scan!
   - You can jump directly to the **Explore & View** section to browse your collection.
3. Click **Scan & Analyze** to execute a full fresh scan of any incoming photos, identify new duplicates, screen text forwards, detect rapid bursts, and run neural face clustering.

---

### 3.2 Curate & Organize Section (Runtime Actions)
- **Options & EXIF**: Adjust perceptual threshold ($0-64$), worker threads, enhancement mode (`balanced`, `fast`, `best`), and trigger one-click **EXIF Date Back-Filling** for WhatsApp/scanned photos.
- **Duplicates**: Visual comparison of identical and near-identical photos with green **Keeper** badges and 1-click keeper switching.
- **Triage Queue**: Review text-heavy message forwards, receipts, and memes with 1-click "Keep as Photo" or "Move to Junk".
- **Burst Shots**: Review rapid-fire photo sequences ($\le 3$s) with Laplacian sharpness scores, dynamic range scores, and one-click **Keep Best Only** triage.
- **Organize Now**: Moves kept photos into `YYYY/YYYY-MM/`, standalone videos into `video/YYYY/YYYY-MM/`, duplicates to `to-be-deleted/`, and automatically updates your People, Flashbacks, and History with the new file locations!

---

### 3.3 Explore & View Section (Library Browsing)
- **People Albums**: Powered by OpenCV YuNet neural face detection and SFace 128-dimensional biometric embeddings. Clusters photos by genuine facial geometry (not color). Supports inline renaming and person photo galleries.
- **Flashbacks**: "On This Day" anniversary memory cards showcasing photos taken on today's calendar date in previous years.
- **World Map**: Interactive Leaflet GPS map with reverse-geocoded city and country marker clusters.
- **AI Semantic Search**: Offline natural language query engine searching by colors, diurnal phases (`sunset`, `golden hour`), seasons (`winter`, `summer`), and locations.
- **History & Rollback**: Complete SQLite audit trail of every past run with single-click instant rollback (`--undo`).


### 3.5 Tab 4: AI Semantic Search
- 100% offline search using local visual and temporal heuristics.
- Type natural language queries in the search box:
  - By time/lighting: `sunset`, `golden hour`, `night`, `afternoon`.
  - By season: `winter`, `summer`, `autumn`, `spring`.
  - By color: `red`, `blue`, `green`, `yellow`, `black`, `white`.
  - By location: `Paris`, `Tokyo`, `New York`, `London`.
  - Combined queries: `winter sunset red`.
- Click on any search result card to preview the full image.

---

### 3.6 Tab 5: World Map
- Interactive Leaflet.js world map rendering geotagged photos.
- Clustered pins indicate photo counts in specific regions.
- Click any marker to view:
  - Full-resolution thumbnail.
  - Reverse-geocoded City and Country (resolved via offline KD-tree).
  - Exact GPS coordinates and date taken.

---

### 3.7 Tab 6: Burst Shots & Best Shot Selector
- Groups photos taken within 3 seconds of each other with visual similarity.
- Displays calculated scores for each shot:
  - **Sharpness**: Laplacian variance.
  - **Exposure**: Dynamic range and exposure balance.
- The highest-scoring photo is highlighted with a **Best Shot** badge.
- Click **Keep Best Only** to mark all suboptimal burst frames for movement to `to-be-deleted/`.

---

### 3.8 Tab 7: People Albums
- Automatically clusters recurring faces using local Haar detection and HSV spatial color vectors.
- Features:
  - High-resolution cropped face avatars.
  - Photo count badges.
  - **Inline Renaming**: Click the pencil icon to rename `"Person 1"` to a custom name (e.g. `"Mom"`, `"Alex"`).
  - Click any person card to open a modal gallery displaying all photos containing that individual.

---

### 3.9 Tab 8: Flashbacks ("On This Day")
- Scans your library for memories taken on **today's calendar date** in previous years (e.g., exactly 1 year ago, 3 years ago, 5 years ago).
- Displays anniversary cards with year badges, date taken, and image previews.

---

### 3.10 Tab 9: History & Rollback
- Displays all past organization runs recorded in `.photo_organizer_journal.db`.
- Details recorded:
  - Run ID and timestamp.
  - Number of files moved and directories removed.
  - Run status (`COMPLETED`, `ROLLED_BACK`).
- Click **Rollback Run** to trigger an instant restoration of all files to their pre-run directories.

---

### 3.11 Live Photo Companion Video Player
- For iPhone Live Photos, clicking the **Play Live Photo** button on any thumbnail opens a companion modal playing the corresponding `.MOV` video clip side-by-side with the high-resolution `.HEIC` still frame.

---

## 4. Standard Operating Procedures (SOPs)

### SOP 1: Ingesting a WhatsApp Media Dump
WhatsApp strips EXIF metadata and introduces duplicates, memes, and receipts.

1. Launch Web UI:
   ```bash
   python photo_organizer.py --ui /path/to/WhatsApp_Media
   ```
2. In **Options**, click **Backfill EXIF Dates**.
   - The engine parses filenames (e.g. `IMG-20230514-WA0002.jpg`) and injects valid `DateTimeOriginal` headers.
3. Switch to **Duplicates** to verify auto-selected keepers.
4. Switch to **Triage** to inspect borderline memes/screenshots.
5. Click **Organize Library**.
   - Result: Personal photos sorted into `YYYY/YYYY-MM/`, duplicates in `to-be-deleted/`, and memes quarantined in `non-relevant/`.

---

### SOP 2: Organizing Apple Photos / iPhone Exports
iPhone exports contain `.HEIC` images, paired Live Photo `.MOV` videos, and `.AAE` edit sidecars.

1. Ensure `pillow-heif` is installed (included in `requirements.txt`).
2. Run standard organization:
   ```bash
   python photo_organizer.py /path/to/iPhone_Export
   ```
3. Verification:
   - For every `IMG_1234.HEIC`, its companion `IMG_1234.MOV` and `IMG_1234.AAE` are moved into the same destination subfolder (`YYYY/YYYY-MM/`).
   - In the Web UI, open the photo to view the still image and play the companion video.

---

### SOP 3: Organizing DSLR / Mirrorless RAW Collections
DSLR cards contain large RAW files (`.CR2`, `.NEF`, `.ARW`, `.DNG`) and `.XMP` sidecars.

1. Run the organizer:
   ```bash
   python photo_organizer.py /path/to/RAW_Photos
   ```
2. The engine extracts embedded JPEG previews (`\xFF\xD8\xFF`) directly from RAW headers for instant preview generation without slow sensor debayering.
3. `.XMP` sidecars containing Lightroom edits are kept in lockstep with their source RAW files.

---

### SOP 4: Video Library Organization
Standalone video files are organized into a dedicated hierarchy:

- Input: Unsorted `.mp4`, `.mov`, `.mkv`, `.avi` files.
- Destination: `video/YYYY/YYYY-MM/filename.ext`.
- Verification: Live Photo companion videos remain in `YYYY/YYYY-MM/` alongside their still photos, while standalone videos move to `video/`.

---

## 5. Performance Tuning & Large Libraries (50,000+ Assets)

| Collection Size | Recommended Flags | Estimated Scan Time (SSD) |
| :--- | :--- | :--- |
| **< 5,000 files** | Default settings | $\sim 10 - 20$ seconds |
| **5,000 - 25,000 files** | `--workers 8` | $\sim 1 - 2$ minutes |
| **25,000 - 100,000+ files** | `--workers 8 --no-cache` (first run only) | $\sim 4 - 8$ minutes |

### Cache Management
- The persistent SQLite cache is stored at: `/path/to/photos/.photo_cache.sqlite`.
- Subsequent scans after the initial run execute in seconds because unchanged files (matching size and `mtime`) are loaded directly from SQLite.
- To force a full re-hash of all assets:
  ```bash
  python photo_organizer.py --no-cache /path/to/photos
  ```

---

## 6. Troubleshooting & FAQ

### Q1: The Web UI says "Address already in use" (Port 8080)
Another service is using port 8080. Specify an alternative port:
```bash
python photo_organizer.py --ui --port 8081 /path/to/photos
```

### Q2: Some WhatsApp images are dated today instead of when they were received
This occurs when the filename lacks a timestamp and EXIF tags are missing.
- **Resolution**: In the Web UI **Options** tab, click **Backfill EXIF Dates**. The system will inspect fallback file modification headers and write verified timestamps into the image headers.

### Q3: How do I undo an organization run?
- **Web UI**: Open the **History** tab and click **Rollback Run**.
- **CLI**: Execute:
  ```bash
  python photo_organizer.py --undo /path/to/photos
  ```
All files will be moved back to their exact original locations, and created empty folders will be removed.

### Q4: HEIC photos show as broken thumbnails
Install `pillow-heif`:
```bash
pip install pillow-heif
```
The organizer automatically detects and registers the HEIC decoder upon startup.

---

## 7. Operational Pre-Flight Checklist

Before running live organization on a primary media drive:

- [ ] **Backup**: Confirm your primary drive is backed up (or run `--dry-run` first).
- [ ] **Disk Space**: Ensure at least 1 GB of free disk space on the target drive for journal ledgers and temporary swap files during in-place enhancement.
- [ ] **Permissions**: Confirm write permissions on the target directory and all child files.
- [ ] **Tests Passing**: Run `python -m unittest discover tests` to ensure all system dependencies are functional.
- [ ] **Dry-Run Review**: Review dry-run output to verify expected file movement counts.
