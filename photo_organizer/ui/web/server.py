"""FastAPI Web UI application for interactive photo organization, duplicate diffing, and triage."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from starlette.responses import FileResponse

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

from photo_organizer.core.cache import PhotoCache
from photo_organizer.core.discovery import discover_images, is_image_file, is_inside
from photo_organizer.core.exif_writer import backfill_library_exif_dates
from photo_organizer.core.journal import Journal
from photo_organizer.core.raw_heif import extract_raw_preview, get_live_photo_video_path, is_raw_file
from photo_organizer.organizer import get_default_workers, organize, scan_library
from photo_organizer.vision.semantic import search_photos


class SetDirectoryRequest(BaseModel):
    directory: str


class SelectKeeperRequest(BaseModel):
    group_member_path: str
    keeper_path: str  # Path string or '__KEEP_ALL__'


class TriageDecisionRequest(BaseModel):
    path: str
    decision: str  # 'KEEP', 'DISCARD', 'RESET'


class BurstKeepBestRequest(BaseModel):
    group_id: Optional[str] = None
    best_path: Optional[str] = None
    discard_paths: Optional[list[str]] = None


class PersonRenameRequest(BaseModel):
    person_id: str
    name: str


class ExifBackfillRequest(BaseModel):
    dry_run: bool = False


class ScanRequest(BaseModel):
    include_videos: bool = True
    video_folder: str = "video"
    visual_threshold: int = 5
    skip_non_relevant: bool = False


class ExecuteRequest(BaseModel):
    dry_run: bool = False
    include_videos: bool = True
    video_folder: str = "video"
    visual_threshold: int = 5
    skip_enhance: bool = False
    skip_non_relevant: bool = False
    max_size_increase: float = 20.0
    enhancement_mode: str = "balanced"
    workers: int = 0


class UndoRequest(BaseModel):
    run_id: Optional[str] = None


class AppState:
    def __init__(self, initial_root: Optional[Path] = None) -> None:
        self.root_dir: Optional[Path] = initial_root.resolve() if initial_root else None
        self.is_scanning: bool = False
        self.is_organizing: bool = False
        self.scan_progress: Dict[str, Any] = {"stage": "idle", "percent": 0.0, "details": {}}
        self.scan_data: Optional[Dict[str, Any]] = None
        self.custom_keeper_map: Dict[str, str] = {}
        self.custom_triage_decisions: Dict[str, str] = {}
        self.custom_people_names: Dict[str, str] = {}
        self.last_summary: Optional[Dict[str, Any]] = None


def create_app(default_root: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Photo Organizer Studio", version="2.0.0")
    state = AppState(default_root)
    app.state.app_state = state

    @app.get("/", response_class=HTMLResponse)
    def index():
        return get_dashboard_html()

    @app.get("/api/status")
    def get_status():
        journal_runs = []
        if state.root_dir and state.root_dir.exists():
            journal = Journal(root_dir=state.root_dir)
            journal_runs = journal.list_runs(limit=10)

        exif_count = state.scan_data.get("exif_backfill_count", 0) if state.scan_data else 0

        return {
            "root_dir": str(state.root_dir) if state.root_dir else None,
            "is_scanning": state.is_scanning,
            "is_organizing": state.is_organizing,
            "scan_progress": state.scan_progress,
            "has_scan_data": state.scan_data is not None,
            "exif_backfill_count": exif_count,
            "last_summary": state.last_summary,
            "journal_runs": journal_runs,
        }

    @app.post("/api/set-directory")
    def set_directory(req: SetDirectoryRequest):
        target = Path(req.directory).expanduser().resolve()
        if not target.exists() or not target.is_dir():
            raise HTTPException(status_code=400, detail="Target path is not a valid directory.")
        state.root_dir = target
        state.custom_keeper_map.clear()
        state.custom_triage_decisions.clear()
        state.scan_progress = {"stage": "idle", "percent": 0.0, "details": {}}

        # Instant file count discovery without decoding or hashing
        image_paths = discover_images(
            target,
            [
                target / "to-be-deleted",
                target / "non-relevant",
                target / "maybe-non-relevant",
            ],
            include_videos=True,
        )

        # Re-hydrate previously saved state from PhotoCache if available
        cache = PhotoCache(root_dir=target)
        cached_state = cache.load_studio_state()
        if cached_state:
            state.scan_data = cached_state
            if cached_state.get("custom_people_names"):
                state.custom_people_names.update(cached_state["custom_people_names"])
            if cached_state.get("last_summary"):
                state.last_summary = cached_state["last_summary"]
        else:
            state.scan_data = None

        journal = Journal(root_dir=target)
        journal_runs = journal.list_runs(limit=10)

        people_count = len(state.scan_data.get("people_albums", [])) if state.scan_data else 0
        flashback_count = len(state.scan_data.get("flashbacks", [])) if state.scan_data else 0
        geo_count = len(state.scan_data.get("geo_photos", [])) if state.scan_data else 0
        burst_count = len(state.scan_data.get("burst_groups", [])) if state.scan_data else 0

        return {
            "status": "ok",
            "root_dir": str(state.root_dir),
            "file_count": len(image_paths),
            "has_cached_state": cached_state is not None,
            "people_count": people_count,
            "flashback_count": flashback_count,
            "geo_count": geo_count,
            "burst_count": burst_count,
            "history_count": len(journal_runs),
            "journal_runs": journal_runs,
        }

    def _run_scan_task(root: Path, req: ScanRequest):
        def _on_progress(stage: str, percent: float, details: dict):
            state.scan_progress = {"stage": stage, "percent": round(percent * 100, 1), "details": details}

        try:
            state.is_scanning = True
            _on_progress("Discovering files in library...", 0.02, {})
            workers = get_default_workers()
            result = scan_library(
                root,
                visual_threshold=req.visual_threshold,
                skip_non_relevant=req.skip_non_relevant,
                include_videos=req.include_videos,
                video_folder_name=req.video_folder,
                worker_count=workers,
                enable_cache=True,
                progress_callback=_on_progress,
            )
            state.scan_data = result
            _on_progress("Scan complete.", 1.0, {"ready": True})
        except Exception as exc:
            _on_progress(f"Scan error: {exc}", 0.0, {"error": str(exc)})
        finally:
            state.is_scanning = False

    @app.post("/api/scan")
    def start_scan(req: Optional[ScanRequest] = None, background_tasks: BackgroundTasks = BackgroundTasks()):
        if not state.root_dir or not state.root_dir.exists():
            raise HTTPException(status_code=400, detail="Please select a valid directory first.")
        if state.is_scanning or state.is_organizing:
            raise HTTPException(status_code=409, detail="A task is already running.")

        if req is None:
            req = ScanRequest()

        state.custom_keeper_map.clear()
        state.custom_triage_decisions.clear()
        background_tasks.add_task(_run_scan_task, state.root_dir, req)
        return {"status": "scanning_started"}

    @app.get("/api/duplicates")
    def get_duplicates():
        if not state.scan_data:
            return {"clusters": []}

        clusters = state.scan_data.get("duplicate_clusters", [])
        for cluster in clusters:
            members = cluster["members"]
            first_path = members[0]["path"] if members else None
            active_keeper = state.custom_keeper_map.get(first_path, cluster["keeper_path"])
            for m in members:
                m["is_keeper"] = (m["path"] == active_keeper)
            cluster["active_keeper"] = active_keeper

        return {"clusters": clusters}

    @app.post("/api/duplicates/select-keeper")
    def select_keeper(req: SelectKeeperRequest):
        state.custom_keeper_map[req.group_member_path] = req.keeper_path
        return {"status": "ok", "keeper": req.keeper_path}

    @app.get("/api/triage")
    def get_triage():
        if not state.scan_data:
            return {"non_relevant": [], "maybe_non_relevant": []}

        nr = [dict(item) for item in state.scan_data.get("non_relevant_items", [])]
        maybe = [dict(item) for item in state.scan_data.get("maybe_non_relevant_items", [])]

        for item in nr + maybe:
            resolved_p = str(Path(item["path"]).resolve())
            item["user_decision"] = state.custom_triage_decisions.get(
                item["path"],
                state.custom_triage_decisions.get(resolved_p, "DEFAULT")
            )

        return {"non_relevant": nr, "maybe_non_relevant": maybe}

    @app.post("/api/triage/decide")
    def triage_decide(req: TriageDecisionRequest):
        resolved = str(Path(req.path).resolve())
        if req.decision == "RESET":
            state.custom_triage_decisions.pop(req.path, None)
            state.custom_triage_decisions.pop(resolved, None)
        else:
            state.custom_triage_decisions[req.path] = req.decision
            state.custom_triage_decisions[resolved] = req.decision
        return {"status": "ok", "decision": req.decision}

    def _run_organize_task(req: ExecuteRequest):
        def _on_progress(stage: str, percent: float, details: dict):
            state.scan_progress = {"stage": stage, "percent": round(percent * 100, 1), "details": details}

        try:
            state.is_organizing = True
            workers = req.workers if req.workers > 0 else get_default_workers()
            journal = Journal(root_dir=state.root_dir)
            summary = organize(
                root=state.root_dir,
                dry_run=req.dry_run,
                visual_threshold=req.visual_threshold,
                skip_enhance=req.skip_enhance,
                skip_non_relevant=req.skip_non_relevant,
                include_videos=req.include_videos,
                video_folder_name=req.video_folder,
                max_size_increase_percent=req.max_size_increase,
                enhancement_mode=req.enhancement_mode,
                worker_count=workers,
                enable_cache=True,
                journal=journal,
                progress_callback=_on_progress,
                custom_keeper_map=state.custom_keeper_map,
                custom_triage_decisions=state.custom_triage_decisions,
            )
            state.last_summary = summary.__dict__
            
            # Maintain and update People, Flashbacks, and Map data with new file paths
            if not req.dry_run:
                moved_map = summary.moved_paths_map
                if state.scan_data:
                    if moved_map:
                        for pa in state.scan_data.get("people_albums", []):
                            pa["photo_paths"] = [
                                moved_map.get(str(Path(p).resolve()), p)
                                for p in pa.get("photo_paths", [])
                            ]
                        for fb in state.scan_data.get("flashbacks", []):
                            for item in fb.get("photos", []):
                                if "path" in item:
                                    item["path"] = moved_map.get(str(Path(item["path"]).resolve()), item["path"])
                        for gp in state.scan_data.get("geo_photos", []):
                            if "path" in gp:
                                gp["path"] = moved_map.get(str(Path(gp["path"]).resolve()), gp["path"])

                    # Clear only the resolved runtime curation queues
                    state.scan_data["duplicate_clusters"] = []
                    state.scan_data["non_relevant_items"] = []
                    state.scan_data["maybe_non_relevant_items"] = []
                    state.custom_keeper_map.clear()
                    state.custom_triage_decisions.clear()

                    cache = PhotoCache(root_dir=state.root_dir)
                    cache.save_studio_state(state.scan_data)
                else:
                    cache = PhotoCache(root_dir=state.root_dir)
                    loaded = cache.load_studio_state()
                    if loaded:
                        state.scan_data = loaded
        except Exception as exc:
            _on_progress(f"Organize error: {exc}", 0.0, {"error": str(exc)})
        finally:
            state.is_organizing = False

    @app.post("/api/execute")
    def execute_organize(req: ExecuteRequest, background_tasks: BackgroundTasks):
        if not state.root_dir or not state.root_dir.exists():
            raise HTTPException(status_code=400, detail="Please select a valid directory first.")
        if state.is_scanning or state.is_organizing:
            raise HTTPException(status_code=409, detail="A task is already running.")

        background_tasks.add_task(_run_organize_task, req)
        return {"status": "started", "dry_run": req.dry_run}

    @app.post("/api/undo")
    def undo_action(req: UndoRequest):
        if not state.root_dir or not state.root_dir.exists():
            raise HTTPException(status_code=400, detail="No active folder selected.")

        journal = Journal(root_dir=state.root_dir)
        restored, failed, errors = journal.undo_run(req.run_id)
        return {
            "status": "ok",
            "restored_count": restored,
            "failed_count": failed,
            "errors": errors,
        }

    @app.get("/api/image-preview")
    def image_preview(path: str = Query(...)):
        file_path = Path(path).resolve()
        if not file_path.exists() or not is_image_file(file_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # Security check: must reside inside root_dir if configured
        if state.root_dir and not is_inside(file_path, state.root_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if Image is None:
            return Response(content=file_path.read_bytes(), media_type="image/jpeg")

        try:
            img = None
            if is_raw_file(file_path):
                img = extract_raw_preview(file_path)
            if img is None:
                img = Image.open(file_path)

            thumb = ImageOps.exif_transpose(img).convert("RGB")
            # 400x400 bilinear thumbnail generation is 3x faster and lightweight
            thumb.thumbnail((400, 400), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=75)
            return Response(
                content=buf.getvalue(),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400, immutable"},
            )
        except Exception:
            return Response(
                content=file_path.read_bytes(),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400, immutable"},
            )

    @app.get("/api/search")
    def search_library(q: str = Query("")):
        if not state.scan_data or "photos" not in state.scan_data:
            return {"query": q, "count": 0, "results": []}

        photos = state.scan_data["photos"]
        matches = search_photos(q, photos)
        results = []
        for photo, score in matches[:100]:
            results.append({
                "path": str(photo.path),
                "filename": photo.path.name,
                "score": round(score, 1),
                "taken_at": photo.taken_at.strftime("%Y-%m-%d %H:%M"),
                "city": photo.city,
                "country": photo.country,
                "is_video": photo.is_video,
                "has_live_photo_video": photo.has_live_photo_video,
            })
        return {"query": q, "count": len(results), "results": results}

    @app.get("/api/map")
    def get_map_photos():
        if not state.scan_data:
            return {"geo_photos": []}
        return {"geo_photos": state.scan_data.get("geo_photos", [])}

    @app.get("/api/bursts")
    def get_bursts():
        if not state.scan_data:
            return {"burst_groups": []}
        return {"burst_groups": state.scan_data.get("burst_groups", [])}

    @app.post("/api/bursts/keep-best")
    def burst_keep_best(req: BurstKeepBestRequest):
        if not state.scan_data:
            raise HTTPException(status_code=400, detail="No scan data available.")

        bursts = state.scan_data.get("burst_groups", [])
        updated_count = 0
        if req.discard_paths:
            for p in req.discard_paths:
                state.custom_triage_decisions[p] = "DELETE"
                updated_count += 1
        elif req.group_id:
            for b in bursts:
                if b.get("group_id") == req.group_id:
                    best_p = req.best_path or b.get("best_path")
                    for bp in b.get("photos", []):
                        p = bp.get("path")
                        if p != best_p:
                            state.custom_triage_decisions[p] = "DELETE"
                            updated_count += 1
                    break
        else:
            # Apply to all burst groups
            for b in bursts:
                best_p = b.get("best_path")
                for bp in b.get("photos", []):
                    p = bp.get("path")
                    if p != best_p:
                        state.custom_triage_decisions[p] = "DELETE"
                        updated_count += 1

        return {"status": "ok", "marked_for_deletion": updated_count}

    @app.get("/api/people")
    def get_people():
        if not state.scan_data:
            return {"people": []}
        people = state.scan_data.get("people_albums", [])
        result = []
        for p in people:
            item = dict(p)
            item["name"] = state.custom_people_names.get(item["person_id"], item["name"])
            result.append(item)
        return {"people": result}

    @app.post("/api/people/rename")
    def rename_person(req: PersonRenameRequest):
        state.custom_people_names[req.person_id] = req.name.strip()
        if state.scan_data:
            for p in state.scan_data.get("people_albums", []):
                if p.get("person_id") == req.person_id:
                    p["name"] = req.name.strip()
        return {"status": "ok", "person_id": req.person_id, "name": req.name.strip()}

    @app.get("/api/flashbacks")
    def get_flashbacks():
        if not state.scan_data:
            return {"flashbacks": []}
        return {"flashbacks": state.scan_data.get("flashbacks", [])}

    @app.post("/api/exif/backfill")
    def backfill_exif(req: ExifBackfillRequest):
        if not state.scan_data or "photos" not in state.scan_data:
            raise HTTPException(status_code=400, detail="Scan the library first before backfilling EXIF.")

        photos = state.scan_data["photos"]
        updated_count = backfill_library_exif_dates(photos, dry_run=req.dry_run)
        return {
            "status": "ok",
            "dry_run": req.dry_run,
            "updated_count": updated_count,
        }

    @app.get("/api/live-photo-video")
    def live_photo_video(path: str = Query(...)):
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Photo file not found")
        if state.root_dir and not is_inside(file_path, state.root_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        video_path = get_live_photo_video_path(file_path)
        if not video_path or not video_path.exists():
            raise HTTPException(status_code=404, detail="No companion video found for this photo")

        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            filename=video_path.name,
        )

    return app


def get_dashboard_html() -> str:
    """Returns the single-page responsive Tailwind CSS web dashboard with dedicated Options & Media Rules tab, KPI summary, and video handling support."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Photo Organizer Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body class="bg-slate-50 text-slate-800 font-sans min-h-screen flex flex-col">

  <!-- Header -->
  <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold text-xl shadow-md shadow-indigo-100">
          📸
        </div>
        <div>
          <h1 class="text-lg font-bold text-slate-900 leading-tight">Photo Organizer Studio</h1>
          <p class="text-xs text-slate-500">Autonomous Curation, Deduplication & Quality Enhancement</p>
        </div>
      </div>

      <!-- Quick Option Summary Pills (Clickable shortcuts to Options tab) -->
      <div class="hidden md:flex items-center gap-2">
        <button onclick="switchTab('tab-options')" title="Click to configure enhancement rules"
          class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 hover:bg-indigo-50 hover:text-indigo-700 border border-slate-200 transition-colors">
          <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
          <span id="pill-enhance">Enhance: Balanced (+20%)</span>
        </button>
        <button onclick="switchTab('tab-options')" title="Click to configure video rules"
          class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 border border-slate-200 transition-colors">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span id="pill-videos">Videos: Included</span>
        </button>
        <button onclick="switchTab('tab-options')" title="Click to configure deduplication sensitivity"
          class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 hover:bg-amber-50 hover:text-amber-700 border border-slate-200 transition-colors">
          <span class="w-2 h-2 rounded-full bg-amber-500"></span>
          <span id="pill-dedupe">Dedupe: Strict (5)</span>
        </button>
      </div>

      <div class="flex items-center space-x-2">
        <button onclick="undoLastRun()" id="btn-undo-header" class="hidden px-3 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs">
          ↩ Rollback Last Run
        </button>
        <span id="badge-status" class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
          Ready
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- Directory & Primary Control Bar -->
    <div class="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-4 sm:p-5">
      <div class="flex flex-col lg:flex-row gap-3 items-stretch lg:items-center justify-between">
        <div class="flex-1 flex gap-2">
          <div class="relative flex-1">
            <span class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">📁</span>
            <input type="text" id="input-dir" placeholder="Enter path to picture directory (e.g. ~/Pictures or D:/Photos)..." 
              class="w-full pl-10 pr-3 py-2.5 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all font-mono">
          </div>
          <button onclick="setDirectory()" class="px-5 py-2.5 bg-slate-800 hover:bg-slate-900 text-white text-sm font-semibold rounded-xl transition-all shadow-xs shrink-0">
            Set Folder
          </button>
        </div>
        <div class="flex flex-wrap gap-2 shrink-0">
          <button onclick="startScan()" id="btn-scan" class="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-xl transition-all flex items-center gap-2 shadow-xs">
            🔍 Scan & Analyze
          </button>
          <button onclick="openExecuteModal(true)" class="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold rounded-xl transition-all border border-slate-200/80">
            👁️ Preview (Dry Run)
          </button>
          <button onclick="openExecuteModal(false)" class="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-xl transition-all shadow-xs flex items-center gap-1.5">
            ⚡ Organize Now
          </button>
        </div>
      </div>

      <!-- Live Progress Bar -->
      <div id="progress-container" class="hidden mt-4 pt-4 border-t border-slate-100 space-y-2">
        <div class="flex justify-between text-xs font-semibold text-slate-600">
          <span id="progress-stage" class="flex items-center gap-2">
            <span class="inline-block w-2 h-2 rounded-full bg-indigo-600 animate-ping"></span>
            Initializing...
          </span>
          <span id="progress-percent" class="font-mono text-indigo-700">0%</span>
        </div>
        <div class="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
          <div id="progress-bar" class="bg-indigo-600 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
      </div>
    </div>

    <!-- Metric KPI Summary Cards (6 Cards) -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <div class="bg-white rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Discovered</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-total-files" class="text-xl font-bold text-slate-900">0</span>
          <span class="text-[11px] font-medium text-slate-400">files</span>
        </div>
      </div>
      <div onclick="switchTab('tab-duplicates')" class="bg-white hover:bg-indigo-50/40 cursor-pointer rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between transition-colors">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Duplicates</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-duplicate-clusters" class="text-xl font-bold text-indigo-600">0</span>
          <span class="text-[11px] font-medium text-slate-400">clusters</span>
        </div>
      </div>
      <div onclick="switchTab('tab-triage')" class="bg-white hover:bg-amber-50/40 cursor-pointer rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between transition-colors">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Triage Junk</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-triage-items" class="text-xl font-bold text-amber-600">0</span>
          <span class="text-[11px] font-medium text-slate-400">forwards</span>
        </div>
      </div>
      <div onclick="switchTab('tab-map')" class="bg-white hover:bg-blue-50/40 cursor-pointer rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between transition-colors">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Map Pins</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-geo-photos" class="text-xl font-bold text-blue-600">0</span>
          <span class="text-[11px] font-medium text-slate-400">places</span>
        </div>
      </div>
      <div onclick="switchTab('tab-bursts')" class="bg-white hover:bg-emerald-50/40 cursor-pointer rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between transition-colors">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Burst Shots</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-burst-groups" class="text-xl font-bold text-emerald-600">0</span>
          <span class="text-[11px] font-medium text-slate-400">bursts</span>
        </div>
      </div>
      <div onclick="switchTab('tab-people')" class="bg-white hover:bg-purple-50/40 cursor-pointer rounded-xl border border-slate-200 p-3.5 flex flex-col justify-between transition-colors">
        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">People</span>
        <div class="mt-1 flex items-baseline justify-between">
          <span id="kpi-people-count" class="text-xl font-bold text-purple-600">0</span>
          <span class="text-[11px] font-medium text-slate-400">albums</span>
        </div>
      </div>
    </div>

    <!-- Main Studio Mode Switcher: Curate vs Explore -->
    <div class="bg-white rounded-2xl border border-slate-200 p-2 shadow-xs mt-2">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div class="inline-flex p-1 bg-slate-100 rounded-xl space-x-1">
          <button id="btn-section-curate" onclick="switchStudioSection('curate')" class="px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all shadow-xs bg-white text-indigo-700 flex items-center gap-2">
            <span>⚡ Curate & Organize</span>
            <span class="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700">Actions</span>
          </button>
          <button id="btn-section-explore" onclick="switchStudioSection('explore')" class="px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all text-slate-600 hover:text-slate-900 flex items-center gap-2">
            <span>🧭 Explore & View</span>
            <span class="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-slate-200 text-slate-700">Library</span>
          </button>
        </div>
        <div class="text-xs text-slate-500 px-2 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span id="section-description">Curation Studio: Resolve duplicates, screen forwards, and execute organization</span>
        </div>
      </div>
    </div>

    <!-- SUB-NAVIGATION 1: Curate & Organize Tabs (Runtime actions) -->
    <div id="nav-group-curate" class="border-b border-slate-200 flex flex-wrap gap-y-2 space-x-1 sm:space-x-3">
      <button onclick="switchTab('tab-options')" id="nav-options" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-indigo-600 text-indigo-600 flex items-center gap-1.5">
        <span>⚙️ Options & EXIF</span>
      </button>
      <button onclick="switchTab('tab-duplicates')" id="nav-duplicates" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>🖼️ Duplicates</span>
        <span id="count-duplicates" class="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded-full font-mono">0</span>
      </button>
      <button onclick="switchTab('tab-triage')" id="nav-triage" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>📑 Triage Queue</span>
        <span id="count-triage" class="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded-full font-mono">0</span>
      </button>
      <button onclick="switchTab('tab-bursts')" id="nav-bursts" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>⚡ Burst Shots</span>
        <span id="count-bursts" class="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded-full font-mono">0</span>
      </button>
    </div>

    <!-- SUB-NAVIGATION 2: Explore & View Tabs (Query & Browsing) -->
    <div id="nav-group-explore" class="hidden border-b border-slate-200 flex flex-wrap gap-y-2 space-x-1 sm:space-x-3">
      <button onclick="switchTab('tab-people')" id="nav-people" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>👥 People Albums</span>
        <span id="count-people" class="bg-purple-100 text-purple-700 text-xs px-2 py-0.5 rounded-full font-mono font-bold">0</span>
      </button>
      <button onclick="switchTab('tab-flashbacks')" id="nav-flashbacks" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>✨ Flashbacks</span>
        <span id="count-flashbacks" class="bg-amber-100 text-amber-700 text-xs px-2 py-0.5 rounded-full font-mono font-bold">0</span>
      </button>
      <button onclick="switchTab('tab-map')" id="nav-map" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>🗺️ World Map</span>
        <span id="count-map" class="bg-emerald-100 text-emerald-700 text-xs px-2 py-0.5 rounded-full font-mono font-bold">0</span>
      </button>
      <button onclick="switchTab('tab-search')" id="nav-search" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>🔍 AI Search</span>
        <span id="count-search" class="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded-full font-mono">0</span>
      </button>
      <button onclick="switchTab('tab-history')" id="nav-history" class="tab-btn py-3 px-2 border-b-2 font-semibold text-sm border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-1.5">
        <span>📜 History & Rollback</span>
        <span id="count-history" class="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded-full font-mono">0</span>
      </button>
    </div>

    <!-- TAB 1: Rules & Options -->
    <div id="tab-options" class="tab-content space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

        <!-- Card 1: Adaptive Quality Enhancement Studio -->
        <div class="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div class="flex items-center space-x-2.5">
              <span class="text-xl">🎨</span>
              <div>
                <h3 class="font-bold text-slate-900 text-sm">Adaptive Quality Enhancement Studio</h3>
                <p class="text-xs text-slate-500">Autonomous contrast balancing & detail restoration</p>
              </div>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="cfg-enhance-enabled" checked onchange="syncOptionsToUI()" class="sr-only peer">
              <div class="w-11 h-6 bg-slate-200 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>

          <div class="space-y-4 text-xs">
            <div>
              <label class="block font-semibold text-slate-700 mb-1">Enhancement Strategy Preset</label>
              <select id="cfg-enhance-mode" onchange="syncOptionsToUI()" class="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs bg-slate-50 font-medium">
                <option value="balanced" selected>Balanced (Recommended) — Adaptive S-curve gamma, edge sharpness</option>
                <option value="fast">Fast — Quick re-encode, skips iterative NLMeans denoising</option>
                <option value="best">Best Quality — Deep quality search & NLMeans noise reduction</option>
              </select>
            </div>

            <div>
              <div class="flex justify-between font-semibold text-slate-700 mb-1">
                <span>Max Allowed File Size Growth</span>
                <span id="val-cfg-size" class="text-indigo-600 font-bold font-mono">20%</span>
              </div>
              <input type="range" id="cfg-size-slider" min="0" max="50" step="5" value="20" oninput="syncOptionsToUI()" class="w-full accent-indigo-600 cursor-pointer">
              <p class="text-[11px] text-slate-500 mt-1">If the enhanced image exceeds original size by more than this limit, enhancement is aborted and the original image is preserved 100% untouched to eliminate disk bloating.</p>
            </div>

            <div class="bg-slate-50 rounded-xl p-3 border border-slate-100 space-y-1.5 text-[11px] text-slate-600">
              <div class="flex items-center gap-1.5 font-semibold text-slate-700">
                <span>🛡️ Non-Destructive Protection Rules:</span>
              </div>
              <p>• <strong>EXIF & ICC Preservation:</strong> Original camera metadata, orientation tags, and color profiles are faithfully retained.</p>
              <p>• <strong>Automated Exclusion:</strong> Standalone video files, RAW formats, animated GIFs, and HEIC files are automatically protected from re-compression.</p>
            </div>
          </div>
        </div>

        <!-- Card 2: Video & Multi-Format Media Management -->
        <div class="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div class="flex items-center space-x-2.5">
              <span class="text-xl">🎬</span>
              <div>
                <h3 class="font-bold text-slate-900 text-sm">Dedicated Video Hierarchy & Media Handling</h3>
                <p class="text-xs text-slate-500">Separate video/YYYY/YYYY-MM/ structure & atom dates</p>
              </div>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="cfg-include-videos" checked onchange="syncOptionsToUI()" class="sr-only peer">
              <div class="w-11 h-6 bg-slate-200 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
            </label>
          </div>

          <div class="space-y-3 text-xs">
            <div class="flex items-start gap-2.5 p-2.5 bg-emerald-50/50 border border-emerald-100 rounded-xl">
              <span class="text-emerald-700 font-bold text-sm">✓</span>
              <div>
                <p class="font-bold text-emerald-900">Separate Video Hierarchy (video/YYYY/YYYY-MM/)</p>
                <p class="text-[11px] text-emerald-700 mt-0.5">All video files (<code>.mp4</code>, <code>.mov</code>, <code>.m4v</code>, <code>.avi</code>, <code>.mkv</code>) are organized into a separate <code>video/</code> folder under the Set Folder, structured into identical <code>YYYY/YYYY-MM/</code> monthly timeline subfolders.</p>
              </div>
            </div>

            <div class="flex items-start gap-2.5 p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
              <span class="text-indigo-600 font-bold text-sm">✓</span>
              <div>
                <p class="font-bold text-slate-800">QuickTime mvhd Atom Date Extraction</p>
                <p class="text-[11px] text-slate-600 mt-0.5">Original capture timestamps are extracted from binary <code>mvhd</code> atom headers inside MP4/MOV files (falling back to filename date pattern, then filesystem timestamp).</p>
              </div>
            </div>

            <div class="flex items-start gap-2.5 p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
              <span class="text-amber-600 font-bold text-sm">✓</span>
              <div>
                <p class="font-bold text-slate-800">Zero Transcoding / 100% Lossless Guarantee</p>
                <p class="text-[11px] text-slate-600 mt-0.5">Videos are <strong>never re-encoded or resized</strong>. Standalone videos are moved bit-for-bit lossless, avoiding CPU spikes and keeping video quality 100% pristine.</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Card 3: Deduplication Engine -->
        <div class="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div class="flex items-center space-x-2.5">
              <span class="text-xl">🔍</span>
              <div>
                <h3 class="font-bold text-slate-900 text-sm">Deduplication Engine Settings</h3>
                <p class="text-xs text-slate-500">Visual similarity hash & exact SHA-256 binary matching</p>
              </div>
            </div>
            <span class="px-2 py-0.5 bg-indigo-100 text-indigo-800 text-[10px] font-bold rounded-full uppercase tracking-wider">Dual-Stage</span>
          </div>

          <div class="space-y-4 text-xs">
            <div>
              <div class="flex justify-between font-semibold text-slate-700 mb-1">
                <span>Visual Similarity Threshold (Hamming Distance)</span>
                <span id="val-cfg-threshold" class="text-indigo-600 font-bold font-mono">5</span>
              </div>
              <input type="range" id="cfg-threshold-slider" min="0" max="30" step="1" value="5" oninput="syncOptionsToUI()" class="w-full accent-indigo-600 cursor-pointer">
              <div class="flex justify-between text-[10px] text-slate-400 mt-1">
                <span>0 (Exact Pixels)</span>
                <span class="font-semibold text-slate-600">5 (Strict / Default)</span>
                <span>15 (Lenient / Crops)</span>
                <span>30 (Loose)</span>
              </div>
            </div>

            <div class="bg-slate-50 rounded-xl p-3 border border-slate-100 space-y-1.5 text-[11px] text-slate-600">
              <p>• <strong>Exact Binary Deduplication:</strong> Identical file sizes are verified using chunked SHA-256 hashes, instantly detecting identical photo and video copies without memory bloat.</p>
              <p>• <strong>Auto-Keeper Selection:</strong> The highest-resolution and highest-bitrate copy is automatically marked as keeper. You can easily click "Set as Keeper" or "Keep All" in the Visual Duplicates tab.</p>
            </div>
          </div>
        </div>

        <!-- Card 4: Content Screening (WhatsApp & Memes) -->
        <div class="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div class="flex items-center space-x-2.5">
              <span class="text-xl">🛡️</span>
              <div>
                <h3 class="font-bold text-slate-900 text-sm">Smart Content Screening (WhatsApp & Memes)</h3>
                <p class="text-xs text-slate-500">Separates forwards & jokes with family photo protection</p>
              </div>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="cfg-screening-enabled" checked onchange="syncOptionsToUI()" class="sr-only peer">
              <div class="w-11 h-6 bg-slate-200 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>

          <div class="space-y-3 text-xs">
            <div class="p-2.5 bg-slate-50 border border-slate-200 rounded-xl space-y-1">
              <p class="font-bold text-slate-800">Haar Cascade & HOG Silhouette Protection</p>
              <p class="text-[11px] text-slate-600">Detects faces and human silhouettes. If a photo contains family members standing before a sign or text graphic, it is <strong>protected</strong> and kept in your personal timeline.</p>
            </div>
            <div class="p-2.5 bg-slate-50 border border-slate-200 rounded-xl space-y-1">
              <p class="font-bold text-slate-800">Quarantine Folders for Safe Review</p>
              <p class="text-[11px] text-slate-600">Screened forwards are moved cleanly into <code>non-relevant/</code> and <code>maybe-non-relevant/</code> folders so you can inspect them before any permanent deletion.</p>
            </div>
          </div>
        </div>

        <!-- Card 5: EXIF Date Back-filling (WhatsApp & Scanned Photos) -->
        <div class="md:col-span-2 bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div class="flex items-center space-x-2.5">
              <span class="text-xl">📅</span>
              <div>
                <h3 class="font-bold text-slate-900 text-sm">EXIF Metadata Back-Filling (WhatsApp & Scanned Photos)</h3>
                <p class="text-xs text-slate-500">Permanently write standard EXIF capture timestamps into image headers</p>
              </div>
            </div>
            <span class="px-2.5 py-1 bg-amber-50 text-amber-800 border border-amber-200 text-xs font-bold rounded-full">
              <span id="val-exif-backfill-count">0</span> candidates detected
            </span>
          </div>

          <div class="space-y-3 text-xs">
            <p class="text-slate-600 leading-relaxed">
              Photos received via WhatsApp, Facebook, or flatbed scanners often lack native EXIF <code>DateTimeOriginal</code> headers. When transferred to a new phone or cloud backup, their capture date can revert to today's date!
            </p>
            <p class="text-slate-600 leading-relaxed">
              This tool parses the capture date from the WhatsApp filename (e.g. <code>IMG-20240512-WA0001.jpg</code>) or filesystem timestamp, and writes standard EXIF headers (<code>DateTimeOriginal</code>, <code>DateTimeDigitized</code>, <code>DateTime</code>) directly into the image file non-destructively.
            </p>
            <div class="flex flex-wrap items-center gap-3 pt-2">
              <button onclick="runExifBackfill(true)" class="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-xl border border-slate-300 transition-colors">
                Preview Backfill (Dry Run)
              </button>
              <button onclick="runExifBackfill(false)" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-xl transition-colors shadow-xs">
                Write EXIF Headers Now
              </button>
              <span id="exif-backfill-status" class="text-xs font-semibold text-indigo-700"></span>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 2: Visual Duplicates -->
    <div id="tab-duplicates" class="tab-content hidden space-y-4">
      <div id="duplicates-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">✨</div>
        <h3 class="text-base font-bold text-slate-800">No duplicate clusters loaded</h3>
        <p class="text-sm text-slate-500 mt-1">Click "Scan & Analyze" to discover identical and visual duplicates.</p>
      </div>
      <div id="duplicates-list" class="space-y-4"></div>
      <div id="duplicates-more-container" class="hidden text-center py-3">
        <button onclick="renderMoreDuplicates()" class="px-6 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold rounded-xl border border-slate-300 transition-colors">
          Load More Duplicate Groups (<span id="duplicates-remaining">0</span> remaining)
        </button>
      </div>
    </div>

    <!-- TAB 3: Triage Queue -->
    <div id="tab-triage" class="tab-content hidden space-y-4">
      <!-- Instruction & Batch Action Banner -->
      <div class="bg-white border border-slate-200 rounded-2xl p-4 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-xl shrink-0">
            📑
          </div>
          <div>
            <h3 class="font-bold text-slate-900 text-sm">Smart Content Screening & Triage</h3>
            <p class="text-xs text-slate-500 mt-0.5">
              Review flagged memes, WhatsApp forwards, and screenshots. Decisions are staged and applied when you click <strong>Organize Now</strong>.
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2 self-stretch md:self-auto shrink-0">
          <button onclick="applyAllTriage('KEEP')" class="flex-1 md:flex-none px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 transition-colors shadow-xs">
            Keep All as Photos
          </button>
          <button onclick="applyAllTriage('DISCARD')" class="flex-1 md:flex-none px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 transition-colors shadow-xs">
            Move All to Junk
          </button>
        </div>
      </div>

      <div id="triage-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">📑</div>
        <h3 class="text-base font-bold text-slate-800">No text forwards or memes flagged</h3>
        <p class="text-sm text-slate-500 mt-1">High-confidence WhatsApp forwards and text graphics will appear here for review.</p>
      </div>
      <div id="triage-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
      <div id="triage-more-container" class="hidden text-center py-3">
        <button onclick="renderMoreTriage()" class="px-6 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold rounded-xl border border-slate-300 transition-colors">
          Load More Triage Items (<span id="triage-remaining">0</span> remaining)
        </button>
      </div>
    </div>

    <!-- TAB 4: AI Semantic Search -->
    <div id="tab-search" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-5 space-y-4">
        <div class="flex flex-col sm:flex-row gap-3">
          <div class="relative flex-1">
            <span class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">🔍</span>
            <input type="text" id="input-search" onkeydown="if(event.key==='Enter') triggerSearch()" placeholder="Search library (e.g. 'sunset beach', 'sunny nature', 'winter', 'night', 'video', 'paris')..." 
              class="w-full pl-10 pr-3 py-2.5 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all">
          </div>
          <button onclick="triggerSearch()" class="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-xl transition-all shadow-xs shrink-0">
            Search
          </button>
        </div>

        <!-- Suggestion Chips -->
        <div class="flex flex-wrap items-center gap-2 text-xs">
          <span class="text-slate-400 font-medium">Quick suggestions:</span>
          <button onclick="applySearchChip('sunset')" class="px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200 transition-colors">🌅 Sunset</button>
          <button onclick="applySearchChip('beach')" class="px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 transition-colors">🏖️ Beach</button>
          <button onclick="applySearchChip('nature')" class="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 transition-colors">🌲 Nature</button>
          <button onclick="applySearchChip('sunny')" class="px-2.5 py-1 rounded-full bg-yellow-50 text-yellow-700 hover:bg-yellow-100 border border-yellow-200 transition-colors">☀️ Sunny</button>
          <button onclick="applySearchChip('night')" class="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-300 transition-colors">🌙 Night</button>
          <button onclick="applySearchChip('winter')" class="px-2.5 py-1 rounded-full bg-cyan-50 text-cyan-700 hover:bg-cyan-100 border border-cyan-200 transition-colors">❄️ Winter</button>
          <button onclick="applySearchChip('live photo')" class="px-2.5 py-1 rounded-full bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200 transition-colors">▶ Live Photos</button>
          <button onclick="applySearchChip('video')" class="px-2.5 py-1 rounded-full bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 transition-colors">🎬 Videos</button>
        </div>
      </div>

      <div class="flex items-center justify-between text-xs font-semibold text-slate-600 px-1">
        <span>Found <span id="search-count" class="font-bold text-indigo-600">0</span> matching photos</span>
      </div>

      <div id="search-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">🔍</div>
        <h3 class="text-base font-bold text-slate-800">Search your library offline</h3>
        <p id="search-empty-text" class="text-sm text-slate-500 mt-1">Type natural language queries like colors, seasons, time of day, or places to find photos instantly.</p>
      </div>

      <div id="search-results" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div>
    </div>

    <!-- TAB 5: World Map -->
    <div id="tab-map" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-4 space-y-3">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="font-bold text-slate-900 text-sm">Geographic Photo Distribution</h3>
            <p class="text-xs text-slate-500">Offline reverse geocoded city and country coordinates</p>
          </div>
          <span id="map-photo-count" class="text-xs font-semibold text-slate-600"></span>
        </div>
        <div id="map-city-list" class="flex flex-wrap gap-2 pt-1 max-h-24 overflow-y-auto"></div>
      </div>

      <div id="map-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">🗺️</div>
        <h3 class="text-base font-bold text-slate-800">No geo-tagged photos detected</h3>
        <p class="text-sm text-slate-500 mt-1">Photos with GPS coordinates in EXIF headers will automatically map here.</p>
      </div>

      <div id="map-container" class="w-full h-[520px] rounded-2xl border border-slate-200 overflow-hidden shadow-xs"></div>
    </div>

    <!-- TAB 6: Burst Shots -->
    <div id="tab-bursts" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <h3 class="font-bold text-slate-900 text-sm">Burst Shot Detector & Best Shot Selector</h3>
          <p class="text-xs text-slate-500">Clusters rapid sequences (within 3 seconds) and scores sharpness & exposure</p>
        </div>
        <button onclick="keepBestAllBursts()" class="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white text-xs font-bold rounded-xl transition-all shadow-xs flex items-center gap-1.5 shrink-0">
          ⚡ Auto-Select All Best Shots
        </button>
      </div>

      <div id="bursts-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">⚡</div>
        <h3 class="text-base font-bold text-slate-800">No burst sequences found</h3>
        <p class="text-sm text-slate-500 mt-1">Rapid photo sequences taken within 3 seconds of each other will appear here.</p>
      </div>

      <div id="bursts-list" class="space-y-4"></div>
    </div>

    <!-- TAB 7: People Albums -->
    <div id="tab-people" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-4">
        <h3 class="font-bold text-slate-900 text-sm">People Albums (100% Local Facial Clustering)</h3>
        <p class="text-xs text-slate-500">Fast spatial color & facial feature clustering. Click ✏️ to label names.</p>
      </div>

      <div id="people-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">👥</div>
        <h3 class="text-base font-bold text-slate-800">No faces clustered yet</h3>
        <p class="text-sm text-slate-500 mt-1">Faces detected in your library photos will be grouped into people albums here.</p>
      </div>

      <div id="people-list" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4"></div>
    </div>

    <!-- TAB 8: Flashbacks -->
    <div id="tab-flashbacks" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-4">
        <h3 class="font-bold text-slate-900 text-sm">On This Day & Flashback Memories</h3>
        <p class="text-xs text-slate-500">Relive moments captured on this day in past years</p>
      </div>

      <div id="flashbacks-empty" class="text-center py-16 bg-white rounded-2xl border border-slate-200">
        <div class="text-4xl mb-3">✨</div>
        <h3 class="text-base font-bold text-slate-800">No flashback memories for today</h3>
        <p class="text-sm text-slate-500 mt-1">Photos taken on today's calendar date across previous years will automatically display here.</p>
      </div>

      <div id="flashbacks-list" class="space-y-6"></div>
    </div>

    <!-- TAB 9: Run History & Undo Ledger -->
    <div id="tab-history" class="tab-content hidden space-y-4">
      <div class="bg-white rounded-2xl border border-slate-200 p-6 space-y-4">
        <div>
          <h3 class="text-base font-bold text-slate-900">Transaction History & Rollback Ledger</h3>
          <p class="text-xs text-slate-500 mt-0.5">Every file move and enhancement is recorded in the SQLite journal. You can safely rollback any run at any time.</p>
        </div>
        <div id="history-table" class="divide-y divide-slate-100 text-sm"></div>
      </div>
    </div>

  </main>

  <!-- Execute Confirmation Modal -->
  <div id="modal-execute" class="fixed inset-0 bg-slate-900/50 backdrop-blur-xs hidden z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-md w-full p-6 space-y-5">
      <div class="flex items-center justify-between border-b pb-3">
        <div>
          <h3 id="modal-title" class="text-base font-bold text-slate-900">Run Organization</h3>
          <p id="modal-subtitle" class="text-xs text-slate-500">Confirm execution settings</p>
        </div>
        <button onclick="closeExecuteModal()" class="text-slate-400 hover:text-slate-600 text-lg">✕</button>
      </div>

      <div class="space-y-4 text-xs">
        <div>
          <label class="block font-semibold text-slate-700 mb-1">Visual Similarity Threshold (0-30)</label>
          <input type="number" id="opt-threshold" value="5" min="0" max="30" class="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm">
          <p class="text-[11px] text-slate-500 mt-0.5">Lower is stricter. Default: 5.</p>
        </div>
        <div>
          <label class="block font-semibold text-slate-700 mb-1">Max File Size Increase Percentage (%)</label>
          <input type="number" id="opt-size-increase" value="20" min="0" max="100" class="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm">
          <p class="text-[11px] text-slate-500 mt-0.5">Rejects enhancement if output file size exceeds this threshold.</p>
        </div>
        <div>
          <label class="block font-semibold text-slate-700 mb-1">Enhancement Strategy Preset</label>
          <select id="opt-mode" class="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-xs bg-white font-medium">
            <option value="balanced" selected>Balanced (Adaptive contrast & gamma)</option>
            <option value="fast">Fast (Quick re-encode, skips NLMeans)</option>
            <option value="best">Best Quality (Deep iterative search & NLMeans)</option>
          </select>
        </div>

        <div class="space-y-2 pt-2 border-t border-slate-100 font-medium">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" id="opt-include-videos" class="rounded-sm text-emerald-600">
            <span>Organize videos into separate video/YYYY/YYYY-MM/ folder</span>
          </label>
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" id="opt-skip-enhance" class="rounded-sm text-indigo-600">
            <span>Skip image enhancement (organize & dedupe only)</span>
          </label>
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" id="opt-skip-non-relevant" class="rounded-sm text-indigo-600">
            <span>Skip text/forward screening</span>
          </label>
        </div>
      </div>

      <div class="flex justify-end gap-2 pt-3 border-t border-slate-100">
        <button onclick="closeExecuteModal()" class="px-4 py-2 border border-slate-200 rounded-xl text-xs font-semibold hover:bg-slate-50 transition-colors">Cancel</button>
        <button onclick="confirmExecute()" id="modal-submit-btn" class="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors">Confirm & Run</button>
      </div>
    </div>
  </div>

  <!-- Live Photo Video Player Modal -->
  <div id="modal-live-photo" class="fixed inset-0 bg-slate-900/60 backdrop-blur-xs hidden z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-2xl w-full p-4 space-y-3 shadow-xl">
      <div class="flex items-center justify-between pb-2 border-b border-slate-100">
        <h3 class="font-bold text-slate-900 text-sm flex items-center gap-2">
          <span>▶ Live Photo Companion Video</span>
        </h3>
        <button onclick="closeLivePhotoModal()" class="text-slate-400 hover:text-slate-600 font-bold text-xl leading-none">&times;</button>
      </div>
      <div class="aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center">
        <video id="live-photo-video-element" controls autoplay loop class="max-h-full max-w-full"></video>
      </div>
    </div>
  </div>

  <!-- Person Photo Gallery Modal -->
  <div id="modal-person-gallery" class="fixed inset-0 bg-slate-900/60 backdrop-blur-xs hidden z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-4xl w-full p-5 space-y-4 shadow-xl max-h-[85vh] flex flex-col">
      <div class="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <h3 id="gallery-person-title" class="font-bold text-slate-900 text-base">Person Album</h3>
          <p id="gallery-person-subtitle" class="text-xs text-slate-500">Appearing photos</p>
        </div>
        <button onclick="closePersonGallery()" class="text-slate-400 hover:text-slate-600 font-bold text-xl leading-none">&times;</button>
      </div>
      <div id="gallery-person-grid" class="flex-1 overflow-y-auto grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-3 p-1"></div>
    </div>
  </div>

  <script>
    let currentDryRun = false;
    let pollInterval = null;
    let allClusters = [];
    let displayedClustersCount = 0;
    let allTriage = [];
    let displayedTriageCount = 0;
    let allBursts = [];
    let allPeople = [];
    let allFlashbacks = [];
    let mapInstance = null;
    let mapMarkers = [];
    const PAGE_SIZE = 30;

    function syncOptionsToUI() {
      const enhance = document.getElementById('cfg-enhance-enabled').checked;
      const mode = document.getElementById('cfg-enhance-mode').value;
      const sizeLimit = document.getElementById('cfg-size-slider').value;
      const videos = document.getElementById('cfg-include-videos').checked;
      const thresh = document.getElementById('cfg-threshold-slider').value;
      const screening = document.getElementById('cfg-screening-enabled').checked;

      document.getElementById('val-cfg-size').textContent = sizeLimit + "%";
      document.getElementById('val-cfg-threshold').textContent = thresh;

      // Update Header Pills
      document.getElementById('pill-enhance').textContent = enhance ? `Enhance: ${mode.charAt(0).toUpperCase() + mode.slice(1)} (+${sizeLimit}%)` : "Enhance: Off";
      document.getElementById('pill-videos').textContent = videos ? "Videos: in video/YYYY/YYYY-MM/" : "Videos: Skipped";
      document.getElementById('pill-dedupe').textContent = `Dedupe: ${thresh <= 5 ? 'Strict' : 'Lenient'} (${thresh})`;
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.root_dir && !document.getElementById('input-dir').value) {
          document.getElementById('input-dir').value = data.root_dir;
        }

        const badge = document.getElementById('badge-status');
        const progressBar = document.getElementById('progress-bar');
        const progressContainer = document.getElementById('progress-container');
        const progressStage = document.getElementById('progress-stage');
        const progressPercent = document.getElementById('progress-percent');

        if (data.is_scanning || data.is_organizing) {
          badge.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 animate-pulse";
          badge.textContent = data.is_scanning ? "Scanning..." : "Organizing...";
          progressContainer.classList.remove('hidden');
          progressStage.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-amber-500 animate-ping"></span> ${data.scan_progress.stage || "Processing..."}`;
          const pct = data.scan_progress.percent || 0;
          progressPercent.textContent = pct + "%";
          progressBar.style.width = pct + "%";
        } else {
          badge.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800";
          badge.textContent = "Ready";
          if (data.scan_progress && data.scan_progress.percent >= 100) {
            setTimeout(() => progressContainer.classList.add('hidden'), 2500);
          }
        }

        if (data.exif_backfill_count !== undefined) {
          document.getElementById('val-exif-backfill-count').textContent = data.exif_backfill_count;
        }

        if (data.journal_runs && data.journal_runs.length > 0) {
          document.getElementById('btn-undo-header').classList.remove('hidden');
          renderHistory(data.journal_runs);
        }
      } catch (e) {
        console.error("Status fetch failed:", e);
      }
    }

    async function setDirectory() {
      const dir = document.getElementById('input-dir').value.trim();
      if (!dir) return;
      const res = await fetch('/api/set-directory', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({directory: dir})
      });
      if (res.ok) {
        const data = await res.json();
        document.getElementById('kpi-total-files').textContent = data.file_count.toLocaleString();
        const badge = document.getElementById('badge-status');
        badge.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800";
        badge.textContent = `${data.file_count.toLocaleString()} media ready`;

        const progressContainer = document.getElementById('progress-container');
        const progressStage = document.getElementById('progress-stage');
        const progressPercent = document.getElementById('progress-percent');
        const progressBar = document.getElementById('progress-bar');

        progressContainer.classList.remove('hidden');
        progressPercent.textContent = "Ready";
        progressBar.style.width = "0%";

        if (data.has_cached_state || data.history_count > 0) {
          // Pre-populate viewing metrics from previously executed scan or organize
          document.getElementById('kpi-people-count').textContent = data.people_count || 0;
          document.getElementById('kpi-geo-photos').textContent = data.geo_count || 0;
          document.getElementById('kpi-burst-groups').textContent = data.burst_count || 0;
          if (document.getElementById('count-people')) document.getElementById('count-people').textContent = data.people_count || 0;
          if (document.getElementById('count-map')) document.getElementById('count-map').textContent = data.geo_count || 0;
          if (document.getElementById('count-flashbacks')) document.getElementById('count-flashbacks').textContent = data.flashback_count || 0;
          if (document.getElementById('count-history')) document.getElementById('count-history').textContent = data.history_count || 0;
          
          await fetchStatus();
          await loadPeople();
          await loadFlashbacks();
          await fetchMapPins();

          progressStage.innerHTML = `Loaded previous library state: <b>${data.people_count}</b> People albums, <b>${data.flashback_count}</b> Flashback memories, <b>${data.geo_count}</b> Map pins ready.`;
          
          // If library has previously organized/scanned data, switch to Explore view immediately
          if (data.people_count > 0 || data.flashback_count > 0 || data.history_count > 0) {
            switchStudioSection('explore');
          }
        } else {
          progressStage.innerHTML = `Folder set: ${data.file_count.toLocaleString()} media files found. Click "🔍 Scan & Analyze" to begin.`;
        }
      } else {
        alert("Invalid directory path. Please check the path and try again.");
      }
    }

    async function startScan() {
      switchStudioSection('curate');
      const payload = {
        include_videos: document.getElementById('cfg-include-videos').checked,
        video_folder: "video",
        visual_threshold: parseInt(document.getElementById('cfg-threshold-slider').value) || 5,
        skip_non_relevant: !document.getElementById('cfg-screening-enabled').checked
      };

      const res = await fetch('/api/scan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        document.getElementById('progress-container').classList.remove('hidden');
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
          await fetchStatus();
          const sRes = await fetch('/api/status');
          const sData = await sRes.json();
          if (!sData.is_scanning && !sData.is_organizing) {
            clearInterval(pollInterval);
            loadScanResults(true);
          }
        }, 500);
      }
    }

    async function loadScanResults(autoSwitchTab = false) {
      // 1. Duplicates
      const dupRes = await fetch('/api/duplicates');
      const dupData = await dupRes.json();
      allClusters = dupData.clusters || [];
      displayedClustersCount = 0;
      document.getElementById('count-duplicates').textContent = allClusters.length;
      document.getElementById('kpi-duplicate-clusters').textContent = allClusters.length;
      document.getElementById('duplicates-list').innerHTML = '';
      renderMoreDuplicates();

      // 2. Triage
      const trRes = await fetch('/api/triage');
      const trData = await trRes.json();
      const nr = trData.non_relevant || [];
      const maybe = trData.maybe_non_relevant || [];
      allTriage = [
        ...maybe.map(x => ({...x, type: 'Maybe Non-relevant'})),
        ...nr.map(x => ({...x, type: 'High-Confidence Forward'}))
      ];
      displayedTriageCount = 0;
      document.getElementById('count-triage').textContent = allTriage.length;
      document.getElementById('kpi-triage-items').textContent = allTriage.length;
      document.getElementById('triage-list').innerHTML = '';
      renderMoreTriage();

      // 3. Map
      const mapRes = await fetch('/api/map');
      const mapData = await mapRes.json();
      const geoPhotos = mapData.geo_photos || [];
      document.getElementById('count-map').textContent = geoPhotos.length;
      document.getElementById('kpi-geo-photos').textContent = geoPhotos.length;
      renderMap(geoPhotos);

      // 4. Bursts
      await loadBursts();

      // 5. People
      await loadPeople();

      // 6. Flashbacks
      await loadFlashbacks();

      // If requested (e.g. after fresh initial scan), switch to duplicates or triage tab
      if (autoSwitchTab) {
        if (allClusters.length > 0) {
          switchTab('tab-duplicates');
        } else if (allTriage.length > 0) {
          switchTab('tab-triage');
        }
      }
    }

    function isVideoExtension(filePath) {
      return /\\.(mp4|mov|m4v|avi|mkv)$/i.test(filePath);
    }

    function renderMoreDuplicates() {
      const emptyState = document.getElementById('duplicates-empty');
      const container = document.getElementById('duplicates-list');
      const moreContainer = document.getElementById('duplicates-more-container');
      const remainingSpan = document.getElementById('duplicates-remaining');

      if (allClusters.length === 0) {
        emptyState.classList.remove('hidden');
        moreContainer.classList.add('hidden');
        return;
      }
      emptyState.classList.add('hidden');

      const nextBatch = allClusters.slice(displayedClustersCount, displayedClustersCount + PAGE_SIZE);
      displayedClustersCount += nextBatch.length;

      nextBatch.forEach((cluster, idx) => {
        const clusterIndex = displayedClustersCount - nextBatch.length + idx;
        const card = document.createElement('div');
        card.className = "bg-white rounded-2xl border border-slate-200 shadow-xs p-4 space-y-3";

        let membersHtml = '';
        cluster.members.forEach(m => {
          const isKeeper = m.is_keeper;
          const isVideo = isVideoExtension(m.path);
          const borderClass = isKeeper ? "border-emerald-500 bg-emerald-50/30 ring-2 ring-emerald-500/20" : "border-slate-200 bg-white";
          const badgeClass = isKeeper ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-700";

          let previewMediaHtml = '';
          if (isVideo) {
            previewMediaHtml = `
              <div class="relative aspect-video rounded-lg overflow-hidden bg-slate-900 flex flex-col items-center justify-center text-slate-300">
                <span class="text-3xl mb-1">🎬</span>
                <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Video File</span>
                <span class="absolute top-2 left-2 px-2 py-0.5 rounded text-[11px] font-bold ${badgeClass}">
                  ${isKeeper ? "★ KEEP" : "MOVE TO DELETE"}
                </span>
                <span class="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-600 text-white">Lossless</span>
              </div>
            `;
          } else {
            previewMediaHtml = `
              <div class="relative aspect-video rounded-lg overflow-hidden bg-slate-100 flex items-center justify-center">
                <img src="/api/image-preview?path=${encodeURIComponent(m.path)}" loading="lazy" class="object-cover w-full h-full">
                <span class="absolute top-2 left-2 px-2 py-0.5 rounded text-[11px] font-bold ${badgeClass}">
                  ${isKeeper ? "★ KEEP" : "MOVE TO DELETE"}
                </span>
                ${m.companion_count > 0 ? `<span class="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-600 text-white">Live Photo / Sidecar</span>` : ''}
              </div>
            `;
          }

          membersHtml += `
            <div class="flex-1 min-w-[260px] border rounded-xl p-3 ${borderClass} flex flex-col justify-between space-y-2">
              ${previewMediaHtml}
              <div class="text-xs space-y-1">
                <p class="font-semibold text-slate-800 truncate" title="${m.path}">${m.relative_path}</p>
                <div class="flex justify-between text-slate-500 text-[11px]">
                  <span>${m.width > 0 ? m.width + ' × ' + m.height + ' px' : (isVideo ? 'Video stream' : 'Original resolution')}</span>
                  <span>${(m.file_size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
              <button onclick="selectKeeper('${encodeURIComponent(cluster.members[0].path)}', '${encodeURIComponent(m.path)}')"
                class="w-full py-1.5 text-xs font-semibold rounded-lg transition-colors ${isKeeper ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'}">
                ${isKeeper ? 'Selected as Keeper' : 'Set as Keeper'}
              </button>
            </div>
          `;
        });

        card.innerHTML = `
          <div class="flex items-center justify-between pb-2 border-b border-slate-100 text-xs font-semibold text-slate-700">
            <span>Duplicate Group #${clusterIndex + 1} (${cluster.members.length} copies)</span>
            <button onclick="selectKeeper('${encodeURIComponent(cluster.members[0].path)}', '__KEEP_ALL__')" class="text-indigo-600 hover:text-indigo-800 hover:underline">
              Keep All (Do Not Move Any)
            </button>
          </div>
          <div class="flex flex-wrap gap-3">
            ${membersHtml}
          </div>
        `;
        container.appendChild(card);
      });

      const remaining = allClusters.length - displayedClustersCount;
      if (remaining > 0) {
        moreContainer.classList.remove('hidden');
        remainingSpan.textContent = remaining;
      } else {
        moreContainer.classList.add('hidden');
      }
    }

    async function selectKeeper(groupRef, keeperPath) {
      await fetch('/api/duplicates/select-keeper', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          group_member_path: decodeURIComponent(groupRef),
          keeper_path: decodeURIComponent(keeperPath)
        })
      });
      loadScanResults(false);
    }

    function renderTriageCardContent(card, item) {
      const isMaybe = item.type === 'Maybe Non-relevant';
      const tagClass = isMaybe ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-800';
      const decision = item.user_decision || 'DEFAULT';

      let decisionBadge = '';
      let keepBtnClass = 'bg-emerald-50 hover:bg-emerald-100 text-emerald-700 font-semibold';
      let keepBtnText = 'Keep as Photo';
      let junkBtnClass = 'bg-rose-50 hover:bg-rose-100 text-rose-700 font-semibold';
      let junkBtnText = 'Move to Junk';

      if (decision === 'KEEP') {
        card.className = "bg-emerald-50/40 border-2 border-emerald-400 rounded-2xl p-3 shadow-xs flex flex-col justify-between space-y-2 transition-all";
        keepBtnClass = 'bg-emerald-600 text-white font-bold shadow-xs';
        keepBtnText = '✓ Kept as Photo';
        junkBtnClass = 'bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs';
        junkBtnText = 'Move to Junk';
        decisionBadge = `
          <div class="px-2.5 py-1 bg-emerald-100/80 border border-emerald-300 text-emerald-900 rounded-lg text-[11px] font-semibold flex items-center justify-between">
            <span class="flex items-center gap-1">✓ <span>Will keep in library (<span class="font-mono">YYYY/YYYY-MM</span>)</span></span>
            <button onclick="decideTriage('${encodeURIComponent(item.path)}', 'RESET')" class="text-emerald-700 hover:text-emerald-950 underline text-[10px] ml-2">Reset</button>
          </div>
        `;
      } else if (decision === 'DISCARD') {
        card.className = "bg-rose-50/40 border-2 border-rose-400 rounded-2xl p-3 shadow-xs flex flex-col justify-between space-y-2 transition-all";
        junkBtnClass = 'bg-rose-600 text-white font-bold shadow-xs';
        junkBtnText = '✓ Marked as Junk';
        keepBtnClass = 'bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs';
        keepBtnText = 'Keep as Photo';
        decisionBadge = `
          <div class="px-2.5 py-1 bg-rose-100/80 border border-rose-300 text-rose-900 rounded-lg text-[11px] font-semibold flex items-center justify-between">
            <span class="flex items-center gap-1">🗑️ <span>Will move to <span class="font-mono">non-relevant/</span></span></span>
            <button onclick="decideTriage('${encodeURIComponent(item.path)}', 'RESET')" class="text-rose-700 hover:text-rose-950 underline text-[10px] ml-2">Reset</button>
          </div>
        `;
      } else {
        card.className = "bg-white border border-slate-200 rounded-2xl p-3 shadow-xs flex flex-col justify-between space-y-2 transition-all";
        decisionBadge = `
          <div class="px-2.5 py-1 bg-slate-50 border border-slate-200 text-slate-500 rounded-lg text-[11px] italic">
            Default on organize: ${isMaybe ? 'Quarantine to maybe-non-relevant/' : 'Move to non-relevant/ junk'}
          </div>
        `;
      }

      card.innerHTML = `
        <div class="relative aspect-video rounded-lg overflow-hidden bg-slate-100">
          <img src="/api/image-preview?path=${encodeURIComponent(item.path)}" loading="lazy" class="object-cover w-full h-full">
          <span class="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-bold ${tagClass}">
            ${item.type}
          </span>
          <span class="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900/70 text-white">
            Score: ${Number(item.text_score).toFixed(2)}
          </span>
        </div>
        <div class="text-xs space-y-1">
          <p class="font-semibold text-slate-800 truncate" title="${item.path}">${item.relative_path}</p>
          <p class="text-[11px] text-slate-500 leading-tight">${item.reason}</p>
        </div>
        <div class="space-y-1.5 pt-1">
          <div class="flex gap-1.5">
            <button onclick="decideTriage('${encodeURIComponent(item.path)}', 'KEEP')" class="flex-1 py-1.5 text-xs rounded-lg ${keepBtnClass} transition-colors">
              ${keepBtnText}
            </button>
            <button onclick="decideTriage('${encodeURIComponent(item.path)}', 'DISCARD')" class="flex-1 py-1.5 text-xs rounded-lg ${junkBtnClass} transition-colors">
              ${junkBtnText}
            </button>
          </div>
          ${decisionBadge}
        </div>
      `;
    }

    function renderMoreTriage() {
      const emptyState = document.getElementById('triage-empty');
      const container = document.getElementById('triage-list');
      const moreContainer = document.getElementById('triage-more-container');
      const remainingSpan = document.getElementById('triage-remaining');

      if (allTriage.length === 0) {
        emptyState.classList.remove('hidden');
        moreContainer.classList.add('hidden');
        return;
      }
      emptyState.classList.add('hidden');

      const nextBatch = allTriage.slice(displayedTriageCount, displayedTriageCount + PAGE_SIZE);
      displayedTriageCount += nextBatch.length;

      nextBatch.forEach(item => {
        const card = document.createElement('div');
        card.setAttribute('data-path', encodeURIComponent(item.path));
        renderTriageCardContent(card, item);
        container.appendChild(card);
      });

      const remaining = allTriage.length - displayedTriageCount;
      if (remaining > 0) {
        moreContainer.classList.remove('hidden');
        remainingSpan.textContent = remaining;
      } else {
        moreContainer.classList.add('hidden');
      }
    }

    async function decideTriage(encodedPath, decision) {
      const decodedPath = decodeURIComponent(encodedPath);
      const item = allTriage.find(x => x.path === decodedPath);
      if (item) {
        item.user_decision = decision;
        const cards = document.querySelectorAll(`[data-path="${encodedPath}"]`);
        cards.forEach(card => renderTriageCardContent(card, item));
      }
      try {
        await fetch('/api/triage/decide', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: decodedPath, decision: decision})
        });
      } catch (e) {
        console.error("Failed to record triage decision:", e);
      }
    }

    async function applyAllTriage(decision) {
      if (allTriage.length === 0) return;
      allTriage.forEach(item => {
        item.user_decision = decision;
      });
      const cards = document.querySelectorAll('#triage-list [data-path]');
      cards.forEach(card => {
        const p = decodeURIComponent(card.getAttribute('data-path'));
        const item = allTriage.find(x => x.path === p);
        if (item) renderTriageCardContent(card, item);
      });
      for (const item of allTriage) {
        fetch('/api/triage/decide', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: item.path, decision: decision})
        }).catch(console.error);
      }
    }

    function renderHistory(runs) {
      const container = document.getElementById('history-table');
      container.innerHTML = '';
      runs.forEach(r => {
        let summary = {};
        try { summary = JSON.parse(r.summary_json || '{}'); } catch(e){}
        const row = document.createElement('div');
        row.className = "py-3.5 flex items-center justify-between";
        row.innerHTML = `
          <div>
            <p class="font-bold text-slate-800">${r.run_id} ${r.dry_run ? '<span class="text-xs font-semibold text-amber-600 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded-md ml-1">Dry Run</span>' : ''}</p>
            <p class="text-xs text-slate-500 mt-0.5">Started: ${new Date(r.started_at).toLocaleString()} | Organized: ${summary.organized_moved || 0}, Duplicates: ${summary.duplicates_moved || 0}, Enhanced: ${summary.enhanced || 0}</p>
          </div>
          ${!r.dry_run ? `
            <button onclick="undoRun('${r.run_id}')" class="px-3 py-1.5 text-xs font-bold text-rose-700 hover:bg-rose-50 border border-rose-200 rounded-lg transition-colors shadow-xs">
              Undo This Run
            </button>
          ` : ''}
        `;
        container.appendChild(row);
      });
    }

    async function undoLastRun() {
      if (!confirm("Are you sure you want to rollback the last run? Files will be restored to their original folders.")) return;
      const res = await fetch('/api/undo', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
      const data = await res.json();
      alert(`Restored ${data.restored_count} files! (${data.failed_count} failures)`);
      fetchStatus();
      loadScanResults();
    }

    async function undoRun(runId) {
      if (!confirm(`Are you sure you want to rollback run ${runId}?`)) return;
      const res = await fetch('/api/undo', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({run_id: runId})});
      const data = await res.json();
      alert(`Restored ${data.restored_count} files!`);
      fetchStatus();
      loadScanResults();
    }

    let currentSection = 'curate';
    let currentCurateTab = 'tab-options';
    let currentExploreTab = 'tab-people';

    function switchStudioSection(section) {
      currentSection = section;
      const btnCurate = document.getElementById('btn-section-curate');
      const btnExplore = document.getElementById('btn-section-explore');
      const navCurate = document.getElementById('nav-group-curate');
      const navExplore = document.getElementById('nav-group-explore');
      const desc = document.getElementById('section-description');

      if (section === 'curate') {
        btnCurate.className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all shadow-xs bg-white text-indigo-700 flex items-center gap-2';
        btnExplore.className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all text-slate-600 hover:text-slate-900 flex items-center gap-2';
        navCurate.classList.remove('hidden');
        navExplore.classList.add('hidden');
        desc.textContent = 'Curation Studio: Resolve duplicates, screen forwards, and execute organization';
        switchTab(currentCurateTab);
      } else {
        btnExplore.className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all shadow-xs bg-white text-indigo-700 flex items-center gap-2';
        btnCurate.className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all text-slate-600 hover:text-slate-900 flex items-center gap-2';
        navExplore.classList.remove('hidden');
        navCurate.classList.add('hidden');
        desc.textContent = 'Library Explorer: Query people, relive flashbacks, browse world map, and view rollback history';
        switchTab(currentExploreTab);
      }
    }

    function switchTab(tabId) {
      const curateTabs = ['tab-options', 'tab-duplicates', 'tab-triage', 'tab-bursts'];
      const exploreTabs = ['tab-people', 'tab-flashbacks', 'tab-map', 'tab-search', 'tab-history'];

      if (curateTabs.includes(tabId)) {
        currentCurateTab = tabId;
        if (currentSection !== 'curate') {
          currentSection = 'curate';
          document.getElementById('btn-section-curate').className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all shadow-xs bg-white text-indigo-700 flex items-center gap-2';
          document.getElementById('btn-section-explore').className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all text-slate-600 hover:text-slate-900 flex items-center gap-2';
          document.getElementById('nav-group-curate').classList.remove('hidden');
          document.getElementById('nav-group-explore').classList.add('hidden');
          document.getElementById('section-description').textContent = 'Curation Studio: Resolve duplicates, screen forwards, and execute organization';
        }
      } else if (exploreTabs.includes(tabId)) {
        currentExploreTab = tabId;
        if (currentSection !== 'explore') {
          currentSection = 'explore';
          document.getElementById('btn-section-explore').className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all shadow-xs bg-white text-indigo-700 flex items-center gap-2';
          document.getElementById('btn-section-curate').className = 'px-4 sm:px-5 py-2 text-xs sm:text-sm font-bold rounded-lg transition-all text-slate-600 hover:text-slate-900 flex items-center gap-2';
          document.getElementById('nav-group-explore').classList.remove('hidden');
          document.getElementById('nav-group-curate').classList.add('hidden');
          document.getElementById('section-description').textContent = 'Library Explorer: Query people, relive flashbacks, browse world map, and view rollback history';
        }
      }

      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      const activeContent = document.getElementById(tabId);
      if (activeContent) activeContent.classList.remove('hidden');

      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-indigo-600', 'text-indigo-600');
        btn.classList.add('border-transparent', 'text-slate-500');
      });
      const activeBtn = document.getElementById(tabId.replace('tab-', 'nav-'));
      if (activeBtn) {
        activeBtn.classList.remove('border-transparent', 'text-slate-500');
        activeBtn.classList.add('border-indigo-600', 'text-indigo-600');
      }

      if (tabId === 'tab-map' && mapInstance) {
        setTimeout(() => mapInstance.invalidateSize(), 200);
      }
    }

    // --- Search Feature ---
    function applySearchChip(term) {
      document.getElementById('input-search').value = term;
      triggerSearch();
    }

    async function triggerSearch() {
      const q = document.getElementById('input-search').value.trim();
      const container = document.getElementById('search-results');
      const countSpan = document.getElementById('search-count');
      const empty = document.getElementById('search-empty');
      const emptyText = document.getElementById('search-empty-text');

      if (!q) {
        countSpan.textContent = '0';
        container.innerHTML = '';
        empty.classList.remove('hidden');
        emptyText.textContent = "Type natural language queries like colors, seasons, time of day, or places to find photos instantly.";
        return;
      }

      const res = await fetch('/api/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      const results = data.results || [];
      countSpan.textContent = results.length;
      document.getElementById('count-search').textContent = results.length;

      if (results.length === 0) {
        container.innerHTML = '';
        empty.classList.remove('hidden');
        emptyText.textContent = `No matching photos found for "${q}". Try terms like sunset, nature, sunny, night, beach, or video.`;
        return;
      }
      empty.classList.add('hidden');
      container.innerHTML = '';

      results.forEach(photo => {
        const card = document.createElement('div');
        card.className = "bg-white rounded-2xl border border-slate-200 shadow-xs p-3 space-y-2 flex flex-col justify-between";

        let mediaHtml = '';
        if (photo.is_video) {
          mediaHtml = `
            <div class="relative aspect-video rounded-xl overflow-hidden bg-slate-900 flex flex-col items-center justify-center text-slate-300">
              <span class="text-3xl mb-1">🎬</span>
              <span class="text-[10px] font-bold text-slate-400 uppercase">Video</span>
            </div>
          `;
        } else {
          mediaHtml = `
            <div class="relative aspect-video rounded-xl overflow-hidden bg-slate-100 flex items-center justify-center">
              <img src="/api/image-preview?path=${encodeURIComponent(photo.path)}" loading="lazy" class="object-cover w-full h-full">
              ${photo.has_live_photo_video ? `
                <button onclick="playLivePhoto('${encodeURIComponent(photo.path)}')" class="absolute top-2 right-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-1 shadow-md">
                  ▶ Live Photo
                </button>
              ` : ''}
            </div>
          `;
        }

        card.innerHTML = `
          ${mediaHtml}
          <div class="text-xs space-y-1">
            <div class="flex items-center justify-between">
              <p class="font-bold text-slate-800 truncate" title="${photo.filename}">${photo.filename}</p>
              <span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-50 text-indigo-700">★ ${photo.score}</span>
            </div>
            <div class="flex items-center justify-between text-slate-500 text-[11px]">
              <span>${photo.taken_at}</span>
              ${photo.city ? `<span>📍 ${photo.city}</span>` : ''}
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    }

    // --- Map Feature ---
    function renderMap(geoPhotos) {
      const container = document.getElementById('map-container');
      const empty = document.getElementById('map-empty');
      const cityList = document.getElementById('map-city-list');
      const countEl = document.getElementById('map-photo-count');

      if (!geoPhotos || geoPhotos.length === 0) {
        empty.classList.remove('hidden');
        container.classList.add('hidden');
        cityList.innerHTML = '';
        if (countEl) countEl.textContent = '0 photos';
        return;
      }
      empty.classList.add('hidden');
      container.classList.remove('hidden');
      if (countEl) countEl.textContent = `${geoPhotos.length} geo-tagged photos`;

      // Group locations by City, Country
      const cityMap = {};
      geoPhotos.forEach(p => {
        const loc = (p.city || 'Unknown Location') + (p.country ? ', ' + p.country : '');
        if (!cityMap[loc]) cityMap[loc] = [];
        cityMap[loc].push(p);
      });

      cityList.innerHTML = '';
      Object.keys(cityMap).forEach(loc => {
        const photos = cityMap[loc];
        const btn = document.createElement('button');
        btn.className = "px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-200 transition-colors flex items-center gap-1.5";
        btn.innerHTML = `📍 <span>${loc}</span> <span class="bg-indigo-200/70 text-indigo-900 px-1.5 py-0.2 rounded-full text-[10px]">${photos.length}</span>`;
        btn.onclick = () => {
          if (mapInstance && photos[0].lat && photos[0].lon) {
            mapInstance.flyTo([photos[0].lat, photos[0].lon], 12);
          }
        };
        cityList.appendChild(btn);
      });

      if (typeof L === 'undefined') {
        container.innerHTML = `<div class="p-8 text-center text-sm text-slate-500">Leaflet CDN unavailable offline. Found ${geoPhotos.length} photos across ${Object.keys(cityMap).length} locations.</div>`;
        return;
      }

      if (!mapInstance) {
        mapInstance = L.map('map-container').setView([20, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          attribution: '© OpenStreetMap'
        }).addTo(mapInstance);
      }

      mapMarkers.forEach(m => mapInstance.removeLayer(m));
      mapMarkers = [];

      const bounds = [];
      geoPhotos.forEach(p => {
        if (p.lat && p.lon) {
          bounds.push([p.lat, p.lon]);
          const marker = L.marker([p.lat, p.lon]).addTo(mapInstance);
          const popup = `
            <div style="width:160px; font-family:sans-serif;">
              <img src="/api/image-preview?path=${encodeURIComponent(p.path)}" style="width:100%; height:95px; object-fit:cover; border-radius:6px; margin-bottom:4px;">
              <div style="font-size:11px; font-weight:bold; color:#0f172a;">${p.city ? p.city + ', ' + (p.country || '') : 'Location'}</div>
              <div style="font-size:10px; color:#64748b;">${p.taken_at}</div>
            </div>
          `;
          marker.bindPopup(popup);
          mapMarkers.push(marker);
        }
      });

      if (bounds.length > 0) {
        mapInstance.fitBounds(bounds, { padding: [30, 30] });
      }
    }

    // --- Burst Shots Feature ---
    async function loadBursts() {
      const res = await fetch('/api/bursts');
      const data = await res.json();
      allBursts = data.burst_groups || [];
      document.getElementById('count-bursts').textContent = allBursts.length;
      document.getElementById('kpi-burst-groups').textContent = allBursts.length;
      renderBursts();
    }

    function renderBursts() {
      const container = document.getElementById('bursts-list');
      const empty = document.getElementById('bursts-empty');
      container.innerHTML = '';
      if (!allBursts || allBursts.length === 0) {
        empty.classList.remove('hidden');
        return;
      }
      empty.classList.add('hidden');

      allBursts.forEach((group, idx) => {
        const card = document.createElement('div');
        card.className = "bg-white rounded-2xl border border-slate-200 shadow-xs p-4 space-y-3";

        let photosHtml = '';
        group.photos.forEach(bp => {
          const isBest = bp.is_best;
          const cardBorder = isBest ? "border-amber-400 bg-amber-50/40 ring-2 ring-amber-400/30" : "border-slate-200 bg-white";
          photosHtml += `
            <div class="flex-1 min-w-[200px] border rounded-xl p-2.5 ${cardBorder} flex flex-col justify-between space-y-2">
              <div class="relative aspect-square rounded-lg overflow-hidden bg-slate-100 flex items-center justify-center">
                <img src="/api/image-preview?path=${encodeURIComponent(bp.path)}" loading="lazy" class="object-cover w-full h-full">
                ${isBest ? `
                  <span class="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500 text-white shadow-xs flex items-center gap-1">
                    ⭐ BEST SHOT
                  </span>
                ` : ''}
                <span class="absolute bottom-2 right-2 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-slate-900/70 text-white">
                  Sharpness: ${bp.sharpness.toFixed(0)} | Score: ${bp.score.toFixed(1)}
                </span>
              </div>
              <div class="text-[11px] font-medium text-slate-700 truncate" title="${bp.filename}">${bp.filename}</div>
              <button onclick="keepOnlyBestShot('${group.group_id}', '${encodeURIComponent(bp.path)}')"
                class="w-full py-1.5 text-xs font-semibold rounded-lg transition-colors ${isBest ? 'bg-amber-100 text-amber-900 hover:bg-amber-200' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'}">
                ${isBest ? 'Keep Best & Move Rest to Delete' : 'Set as Best & Keep'}
              </button>
            </div>
          `;
        });

        card.innerHTML = `
          <div class="flex items-center justify-between pb-2 border-b border-slate-100 text-xs font-semibold text-slate-700">
            <span class="flex items-center gap-2">
              <span class="text-amber-600 font-bold">⚡ Burst #${idx + 1}</span>
              <span>(${group.photos.length} rapid shots within 3 sec)</span>
            </span>
            <button onclick="keepOnlyBestShot('${group.group_id}', '${encodeURIComponent(group.best_path)}')" class="text-xs font-bold text-amber-700 hover:text-amber-800 bg-amber-50 hover:bg-amber-100 px-2.5 py-1 rounded-lg border border-amber-200 transition-colors">
              ⚡ Keep Best Shot Only
            </button>
          </div>
          <div class="flex flex-wrap gap-3">
            ${photosHtml}
          </div>
        `;
        container.appendChild(card);
      });
    }

    async function keepOnlyBestShot(groupId, bestPathEnc) {
      const bestPath = decodeURIComponent(bestPathEnc);
      const group = allBursts.find(b => b.group_id === groupId);
      if (!group) return;
      const discardPaths = group.photos.filter(p => p.path !== bestPath).map(p => p.path);
      await fetch('/api/bursts/keep-best', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          group_id: groupId,
          best_path: bestPath,
          discard_paths: discardPaths
        })
      });
      group.best_path = bestPath;
      group.photos.forEach(p => { p.is_best = (p.path === bestPath); });
      renderBursts();
    }

    async function keepBestAllBursts() {
      if (!confirm("Select the best shot for all burst groups and mark redundant burst photos for deletion?")) return;
      await fetch('/api/bursts/keep-best', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      alert("Best shots selected for all burst groups! Redundant shots marked to be moved to to-be-deleted upon organization.");
      loadScanResults();
    }

    // --- People Feature ---
    async function loadPeople() {
      const res = await fetch('/api/people');
      const data = await res.json();
      allPeople = data.people || [];
      document.getElementById('count-people').textContent = allPeople.length;
      document.getElementById('kpi-people-count').textContent = allPeople.length;
      renderPeople();
    }

    function renderPeople() {
      const container = document.getElementById('people-list');
      const empty = document.getElementById('people-empty');
      container.innerHTML = '';
      if (!allPeople || allPeople.length === 0) {
        empty.classList.remove('hidden');
        return;
      }
      empty.classList.add('hidden');

      allPeople.forEach(person => {
        const card = document.createElement('div');
        card.className = "bg-white rounded-2xl border border-slate-200 shadow-xs p-4 flex flex-col items-center text-center space-y-3";

        let avatarHtml = '';
        if (person.thumbnail && person.thumbnail.length > 20) {
          avatarHtml = `<img src="data:image/jpeg;base64,${person.thumbnail}" class="w-20 h-20 rounded-full object-cover shadow-sm ring-2 ring-indigo-100">`;
        } else if (person.photo_paths && person.photo_paths.length > 0) {
          avatarHtml = `<img src="/api/image-preview?path=${encodeURIComponent(person.photo_paths[0])}" class="w-20 h-20 rounded-full object-cover shadow-sm ring-2 ring-indigo-100">`;
        } else {
          avatarHtml = `<div class="w-20 h-20 rounded-full bg-indigo-100 flex items-center justify-center text-2xl font-bold text-indigo-600">👤</div>`;
        }

        card.innerHTML = `
          <div class="relative">
            ${avatarHtml}
            <span class="absolute bottom-0 right-0 px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-600 text-white shadow-xs">
              ${person.face_count}
            </span>
          </div>
          <div class="w-full">
            <div class="flex items-center justify-center gap-1.5">
              <h4 id="person-name-${person.person_id}" class="text-sm font-bold text-slate-800 truncate">${person.name}</h4>
              <button onclick="editPersonName('${person.person_id}', '${encodeURIComponent(person.name)}')" title="Rename person" class="text-slate-400 hover:text-indigo-600 text-xs">✏️</button>
            </div>
            <p class="text-[11px] text-slate-500 mt-0.5">${person.face_count} photo appearances</p>
          </div>
          <button onclick="viewPersonGallery('${person.person_id}')" class="w-full py-1.5 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded-xl transition-colors">
            View Album (${person.photo_paths ? person.photo_paths.length : 0})
          </button>
        `;
        container.appendChild(card);
      });
    }

    async function editPersonName(personId, curName) {
      const newName = prompt("Enter name for this person:", decodeURIComponent(curName));
      if (newName && newName.trim()) {
        await fetch('/api/people/rename', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ person_id: personId, name: newName.trim() })
        });
        loadPeople();
      }
    }

    function viewPersonGallery(personId) {
      const person = allPeople.find(p => p.person_id === personId);
      if (!person) return;
      document.getElementById('gallery-person-title').textContent = person.name;
      document.getElementById('gallery-person-subtitle').textContent = `${person.face_count} detected faces across ${(person.photo_paths || []).length} photos`;
      const grid = document.getElementById('gallery-person-grid');
      grid.innerHTML = '';
      (person.photo_paths || []).forEach(p => {
        const item = document.createElement('div');
        item.className = "relative aspect-square rounded-xl overflow-hidden bg-slate-100 border border-slate-200";
        item.innerHTML = `<img src="/api/image-preview?path=${encodeURIComponent(p)}" loading="lazy" class="object-cover w-full h-full">`;
        grid.appendChild(item);
      });
      document.getElementById('modal-person-gallery').classList.remove('hidden');
    }

    function closePersonGallery() {
      document.getElementById('modal-person-gallery').classList.add('hidden');
    }

    // --- Flashbacks Feature ---
    async function loadFlashbacks() {
      const res = await fetch('/api/flashbacks');
      const data = await res.json();
      allFlashbacks = data.flashbacks || [];
      renderFlashbacks();
    }

    function renderFlashbacks() {
      const container = document.getElementById('flashbacks-list');
      const empty = document.getElementById('flashbacks-empty');
      container.innerHTML = '';
      if (!allFlashbacks || allFlashbacks.length === 0) {
        empty.classList.remove('hidden');
        return;
      }
      empty.classList.add('hidden');

      allFlashbacks.forEach(card => {
        const cardEl = document.createElement('div');
        cardEl.className = "bg-white rounded-2xl border border-slate-200 shadow-xs p-5 space-y-3";

        let photosHtml = '';
        (card.photos || []).forEach(p => {
          photosHtml += `
            <div class="relative aspect-square rounded-xl overflow-hidden bg-slate-100 group">
              <img src="/api/image-preview?path=${encodeURIComponent(p.path)}" loading="lazy" class="object-cover w-full h-full group-hover:scale-105 transition-transform duration-300">
              <div class="absolute inset-0 bg-gradient-to-t from-slate-900/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-2 flex flex-col justify-end text-white text-[11px]">
                <p class="font-bold truncate">${p.filename}</p>
                <p class="text-[10px] text-slate-300">${p.taken_at} ${p.city ? '• ' + p.city : ''}</p>
              </div>
            </div>
          `;
        });

        cardEl.innerHTML = `
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <h3 class="font-bold text-slate-900 text-sm flex items-center gap-2">
                <span>✨ ${card.title}</span>
                <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200">${card.date_label}</span>
              </h3>
              <p class="text-xs text-slate-500 mt-0.5">${card.photos.length} captured moments</p>
            </div>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
            ${photosHtml}
          </div>
        `;
        container.appendChild(cardEl);
      });
    }

    // --- Live Photo Video Player ---
    function playLivePhoto(pathEnc) {
      const path = decodeURIComponent(pathEnc);
      const modal = document.getElementById('modal-live-photo');
      const videoEl = document.getElementById('live-photo-video-element');
      videoEl.src = '/api/live-photo-video?path=' + encodeURIComponent(path);
      modal.classList.remove('hidden');
    }

    function closeLivePhotoModal() {
      const modal = document.getElementById('modal-live-photo');
      const videoEl = document.getElementById('live-photo-video-element');
      videoEl.pause();
      videoEl.src = '';
      modal.classList.add('hidden');
    }

    // --- EXIF Back-fill ---
    async function runExifBackfill(dryRun) {
      const statusSpan = document.getElementById('exif-backfill-status');
      statusSpan.textContent = dryRun ? 'Simulating backfill...' : 'Writing EXIF headers...';
      try {
        const res = await fetch('/api/exif/backfill', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ dry_run: dryRun })
        });
        const data = await res.json();
        if (res.ok) {
          statusSpan.textContent = dryRun 
            ? `Dry run: ${data.updated_count} photos can be backfilled.` 
            : `Successfully backfilled ${data.updated_count} photos!`;
        } else {
          statusSpan.textContent = `Error: ${data.detail || 'Failed'}`;
        }
      } catch(e) {
        statusSpan.textContent = `Error: ${e.message}`;
      }
    }

    function openExecuteModal(dryRun) {
      currentDryRun = dryRun;
      document.getElementById('modal-title').textContent = dryRun ? "Run Organization Preview (Dry Run)" : "Execute Organization & Enhancement";
      document.getElementById('modal-subtitle').textContent = dryRun ? "Safe preview: No files will be moved or modified on disk." : "Live execution: Moves files into YYYY/YYYY-MM/ and updates SQLite journal.";
      document.getElementById('modal-submit-btn').textContent = dryRun ? "Run Safe Dry Run" : "Execute Live Now";
      document.getElementById('modal-submit-btn').className = dryRun ? "px-5 py-2 bg-slate-800 hover:bg-slate-900 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors" : "px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors";

      // Pre-fill modal with Options tab values
      document.getElementById('opt-threshold').value = document.getElementById('cfg-threshold-slider').value;
      document.getElementById('opt-size-increase').value = document.getElementById('cfg-size-slider').value;
      document.getElementById('opt-mode').value = document.getElementById('cfg-enhance-mode').value;
      document.getElementById('opt-include-videos').checked = document.getElementById('cfg-include-videos').checked;
      document.getElementById('opt-skip-enhance').checked = !document.getElementById('cfg-enhance-enabled').checked;
      document.getElementById('opt-skip-non-relevant').checked = !document.getElementById('cfg-screening-enabled').checked;

      document.getElementById('modal-execute').classList.remove('hidden');
    }

    function closeExecuteModal() {
      document.getElementById('modal-execute').classList.add('hidden');
    }

    async function confirmExecute() {
      const payload = {
        dry_run: currentDryRun,
        visual_threshold: parseInt(document.getElementById('opt-threshold').value) || 5,
        max_size_increase: parseFloat(document.getElementById('opt-size-increase').value) || 20.0,
        enhancement_mode: document.getElementById('opt-mode').value,
        include_videos: document.getElementById('opt-include-videos').checked,
        video_folder: "video",
        skip_enhance: document.getElementById('opt-skip-enhance').checked,
        skip_non_relevant: document.getElementById('opt-skip-non-relevant').checked,
        workers: 0
      };
      closeExecuteModal();
      const res = await fetch('/api/execute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        document.getElementById('progress-container').classList.remove('hidden');
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
          await fetchStatus();
          const sRes = await fetch('/api/status');
          const sData = await sRes.json();
          if (!sData.is_scanning && !sData.is_organizing) {
            clearInterval(pollInterval);
            await loadScanResults();
            await fetchStatus();
            await loadPeople();
            await loadFlashbacks();
            await fetchMapPins();
            switchStudioSection('explore');
          }
        }, 500);
      }
    }

    // Initialize
    syncOptionsToUI();
    fetchStatus();
  </script>
</body>
</html>
"""
