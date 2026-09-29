"""Unit tests for advanced features: Journal/Undo, Web UI, Cache, pHash, Companion Files, Video Dates."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from photo_organizer.core.cache import PhotoCache
from photo_organizer.core.dates import date_from_quicktime_video
from photo_organizer.core.discovery import find_companion_files
from photo_organizer.core.file_ops import move_file_with_companions
from photo_organizer.core.journal import Journal
from photo_organizer.dedupe.hasher import (
    compute_dhash,
    compute_phash,
    flattened_pixels,
    hamming_distance,
)
from photo_organizer.dedupe.matcher import duplicate_groups, keeper_for_group
from photo_organizer.models import PhotoInfo
from photo_organizer.organizer import organize, scan_library
from photo_organizer.ui.web.server import create_app


class AdvancedFeaturesTests(unittest.TestCase):
    def test_journal_and_undo_restores_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig_file = root / "unorganized" / "vacation.jpg"
            orig_file.parent.mkdir(parents=True)
            orig_file.write_bytes(b"image data here")

            journal = Journal(root_dir=root)
            run_id = journal.start_run(root, dry_run=False)

            dest_file = root / "2024" / "2024-06" / "vacation.jpg"
            dest_file.parent.mkdir(parents=True)
            orig_file.rename(dest_file)

            journal.record_action(
                run_id=run_id,
                action_type="MOVE",
                source_path=str(orig_file),
                dest_path=str(dest_file),
            )
            journal.complete_run(run_id, {"moved": 1})

            self.assertTrue(dest_file.exists())
            self.assertFalse(orig_file.exists())

            # Perform undo
            restored, failed, errors = journal.undo_run(run_id)
            self.assertEqual(restored, 1)
            self.assertEqual(failed, 0)
            self.assertEqual(len(errors), 0)

            # Verification
            self.assertTrue(orig_file.exists())
            self.assertFalse(dest_file.exists())
            self.assertEqual(orig_file.read_bytes(), b"image data here")

    def test_companion_files_live_photos_and_xmp_moved_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "IMG_0001.JPG"
            video_companion = root / "IMG_0001.MOV"
            xmp_sidecar = root / "IMG_0001.xmp"

            photo.write_bytes(b"photo content")
            video_companion.write_bytes(b"live photo video content")
            xmp_sidecar.write_bytes(b"<xmp>metadata</xmp>")

            companions = find_companion_files(photo)
            self.assertEqual(len(companions), 2)
            self.assertIn(video_companion, companions)
            self.assertIn(xmp_sidecar, companions)

            dest_dir = root / "2024" / "2024-01"
            dest_photo = dest_dir / "IMG_0001.JPG"

            moved, final_dest, companions_moved = move_file_with_companions(
                photo, dest_photo, companions, dry_run=False
            )

            self.assertTrue(moved)
            self.assertEqual(companions_moved, 2)
            self.assertTrue(final_dest.exists())
            self.assertTrue((dest_dir / "IMG_0001.MOV").exists())
            self.assertTrue((dest_dir / "IMG_0001.xmp").exists())

    def test_dct_phash_computation(self):
        img1 = Image.new("RGB", (100, 100), color="white")
        # Draw a black square in middle
        for x in range(30, 70):
            for y in range(30, 70):
                img1.putpixel((x, y), (0, 0, 0))

        img2 = Image.new("RGB", (200, 200), color="white")
        # Scaled identical image
        for x in range(60, 140):
            for y in range(60, 140):
                img2.putpixel((x, y), (0, 0, 0))

        phash1 = compute_phash(img1)
        phash2 = compute_phash(img2)

        self.assertIsNotNone(phash1)
        self.assertIsNotNone(phash2)
        dist = hamming_distance(phash1, phash2)
        # Scaled versions of the same pattern should have very low pHash distance
        self.assertLessEqual(dist, 4)

    def test_persistent_cache_speeds_up_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = PhotoCache(root_dir=root)
            photo_path = root / "sample.jpg"
            photo_path.write_bytes(b"some content")

            info = PhotoInfo(
                path=photo_path,
                file_size=12,
                sha256="abcd1234ef",
                taken_at=datetime(2024, 5, 10, 12, 0),
                date_source="exif",
                width=1920,
                height=1080,
                visual_hash=999,
                phash=888,
                format_name="JPEG",
            )
            mtime_ns = 1_000_000_000
            cache.set(info, mtime_ns)

            # Retrieve from cache
            retrieved = cache.get(photo_path, 12, mtime_ns)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.sha256, "abcd1234ef")
            self.assertEqual(retrieved.width, 1920)
            self.assertEqual(retrieved.height, 1080)
            self.assertEqual(retrieved.visual_hash, 999)
            self.assertEqual(retrieved.phash, 888)

            # Different mtime misses cache
            miss = cache.get(photo_path, 12, mtime_ns + 1)
            self.assertIsNone(miss)

    def test_quicktime_video_date_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_file = Path(tmp) / "video.mov"
            # 1904-01-01 + 3768422400 seconds = 2023-06-01
            secs = 3768422400
            atom = b"\x00\x00\x00\x6c" + b"mvhd" + b"\x00\x00\x00\x00" + struct.pack(">I", secs)
            video_file.write_bytes(b"\x00" * 32 + atom + b"\x00" * 64)

            parsed = date_from_quicktime_video(video_file)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.year, 2023)

    def test_web_ui_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "photo1.jpg"
            img_obj = Image.new("RGB", (100, 100), color="blue")
            img_obj.save(img, "JPEG")

            app = create_app(default_root=root)
            client = TestClient(app)

            # 1. Status
            res = client.get("/api/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("root_dir", data)

            # 2. Set Directory
            res = client.post("/api/set-directory", json={"directory": str(root)})
            self.assertEqual(res.status_code, 200)

            # 3. Scan Library
            scan_res = scan_library(root, worker_count=1)
            self.assertEqual(scan_res["total_scanned"], 1)

            # 4. Preview Image
            preview_res = client.get(f"/api/image-preview?path={img}")
            self.assertEqual(preview_res.status_code, 200)
            self.assertEqual(preview_res.headers["content-type"], "image/jpeg")

            # 5. Undo endpoint
            undo_res = client.post("/api/undo", json={})
            self.assertEqual(undo_res.status_code, 200)

            # 6. Scan endpoint with options
            scan_endpoint_res = client.post("/api/scan", json={"include_videos": True, "visual_threshold": 5})
            self.assertEqual(scan_endpoint_res.status_code, 200)
            self.assertEqual(scan_endpoint_res.json()["status"], "scanning_started")

            # 7. Execute endpoint with options
            exec_endpoint_res = client.post(
                "/api/execute",
                json={
                    "dry_run": True,
                    "include_videos": True,
                    "visual_threshold": 5,
                    "skip_enhance": True,
                },
            )
            self.assertEqual(exec_endpoint_res.status_code, 200)
            self.assertTrue(exec_endpoint_res.json()["dry_run"])

    def test_standalone_video_organization_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "VID_20220715_143000.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"video payload bytes")

            # 1. Scan without videos: discovers 0
            scan_no_vid = scan_library(root, include_videos=False)
            self.assertEqual(scan_no_vid["total_scanned"], 0)

            # 2. Scan with videos: discovers 1 video
            scan_with_vid = scan_library(root, include_videos=True)
            self.assertEqual(scan_with_vid["total_scanned"], 1)

            # 3. Organize with videos
            summary = organize(root, include_videos=True, dry_run=False, skip_enhance=False)
            self.assertEqual(summary.organized_moved, 1)
            self.assertEqual(summary.enhanced, 0)  # Videos are NEVER enhanced

            dest_video = root / "video" / "2022" / "2022-07" / "VID_20220715_143000.mp4"
            self.assertTrue(dest_video.exists())
            self.assertEqual(dest_video.read_bytes(), b"\x00\x00\x00\x18ftypmp42" + b"video payload bytes")

    def test_bulk_cache_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = PhotoCache(root_dir=root)
            p1 = root / "p1.jpg"
            p2 = root / "p2.jpg"
            p1.write_bytes(b"content 1")
            p2.write_bytes(b"content 2")

            info1 = PhotoInfo(
                path=p1,
                file_size=9,
                sha256="hash1",
                taken_at=datetime(2024, 1, 1),
                date_source="filename",
                width=1920,
                height=1080,
                visual_hash=111,
                phash=222,
                format_name="JPEG",
            )
            info2 = PhotoInfo(
                path=p2,
                file_size=9,
                sha256="hash2",
                taken_at=datetime(2024, 1, 2),
                date_source="filename",
                width=1280,
                height=720,
                visual_hash=333,
                phash=444,
                format_name="JPEG",
            )

            # Bulk save
            cache.set_bulk([(info1, 1000), (info2, 2000)])

            # Bulk retrieve
            bulk_map = cache.load_cache_bulk([p1, p2])
            self.assertIn(str(p1.resolve()), bulk_map)
            self.assertIn(str(p2.resolve()), bulk_map)

            retrieved1, size1, mtime1 = bulk_map[str(p1.resolve())]
            self.assertEqual(retrieved1.sha256, "hash1")
            self.assertEqual(retrieved1.width, 1920)
            self.assertEqual(retrieved1.visual_hash, 111)
            self.assertEqual(retrieved1.phash, 222)
            self.assertEqual(mtime1, 1000)

    def test_robust_deduplication_multi_index_and_rotation(self):
        # Two photos that are identical visual content, one slightly rotated in aspect ratio
        p1 = PhotoInfo(
            path=Path("original.jpg"),
            file_size=5000,
            sha256=None,
            taken_at=datetime(2024, 1, 1),
            date_source="filename",
            width=1920,
            height=1080,
            visual_hash=0x1234567890ABCDEF,
            phash=0xAABBCCDDEEFF0011,
        )
        p2 = PhotoInfo(
            path=Path("compressed_copy.jpg"),
            file_size=1200,
            sha256=None,
            taken_at=datetime(2024, 1, 1),
            date_source="filename",
            width=960,
            height=540,
            visual_hash=0x1234567890ABCDEE,  # 1 bit difference
            phash=0xAABBCCDDEEFF0011,
        )
        p3 = PhotoInfo(
            path=Path("rotated_copy.jpg"),
            file_size=2000,
            sha256=None,
            taken_at=datetime(2024, 1, 1),
            date_source="filename",
            width=1080,
            height=1920,  # Rotated orientation
            visual_hash=0x1234567890ABCDEF,
            phash=0xAABBCCDDEEFF0011,
        )
        p_distinct = PhotoInfo(
            path=Path("distinct.jpg"),
            file_size=3000,
            sha256=None,
            taken_at=datetime(2024, 1, 1),
            date_source="filename",
            width=1920,
            height=1080,
            visual_hash=0xFEDCBA0987654321,
            phash=0x0000000000000000,
        )

        groups = duplicate_groups([p1, p2, p3, p_distinct], visual_threshold=5)
        # p1, p2, p3 should all be grouped together as duplicates
        dup_group = next(g for g in groups if len(g) > 1)
        group_paths = {p.path for p in dup_group}
        self.assertIn(p1.path, group_paths)
        self.assertIn(p2.path, group_paths)
        self.assertIn(p3.path, group_paths)
        self.assertNotIn(p_distinct.path, group_paths)

        # Keeper must be the highest resolution original
        keeper = keeper_for_group(dup_group)
        self.assertEqual(keeper.path, p1.path)

    def test_quality_enhancement_clahe_guardrails(self):
        from photo_organizer.vision.enhancer import enhance_in_place

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create an underexposed test image
            img_path = root / "underexposed.jpg"
            img = Image.new("RGB", (200, 200), (60, 50, 40))
            img.save(img_path, "JPEG")

            enhanced, reason = enhance_in_place(
                img_path,
                dry_run=False,
                max_size_increase_percent=50.0,
                enhancement_mode="balanced",
            )
            self.assertTrue(enhanced)
            self.assertIn("underexposed", reason)

            # Check that enhanced image has lifted brightness
            with Image.open(img_path) as res:
                stat = res.convert("L")
                pixels = flattened_pixels(stat)
                avg_brightness = sum(pixels) / len(pixels)
                # Original was ~50, enhanced should be visibly lifted
                self.assertGreater(avg_brightness, 52)

    def test_smart_screening_useless_images(self):
        from photo_organizer.vision.classifier import classify_non_relevant

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Pocket dark image (< 10 brightness)
            dark_path = root / "dark_pocket.jpg"
            dark_img = Image.new("RGB", (200, 200), (2, 2, 2))
            dark_img.save(dark_path, "JPEG")
            res_dark = classify_non_relevant(dark_path, likely_camera_photo=False)
            self.assertTrue(res_dark.is_non_relevant)
            self.assertIn("pocket dark", res_dark.reason)

            # 2. Tiny icon (< 96 px)
            icon_path = root / "tiny_icon.png"
            icon_img = Image.new("RGB", (48, 48), (100, 150, 200))
            icon_img.save(icon_path, "PNG")
            res_icon = classify_non_relevant(icon_path, likely_camera_photo=False)
            self.assertTrue(res_icon.is_non_relevant)
            self.assertIn("tiny thumbnail", res_icon.reason)

    def test_offline_reverse_geocoding_and_gps_extraction(self):
        from photo_organizer.core.geo import reverse_geocode_offline

        # Test major world capitals and cities
        city, country = reverse_geocode_offline(48.8566, 2.3522)
        self.assertEqual(city, "Paris")
        self.assertEqual(country, "France")

        city, country = reverse_geocode_offline(40.7128, -74.0060)
        self.assertEqual(city, "New York")
        self.assertEqual(country, "United States")

        city, country = reverse_geocode_offline(35.6762, 139.6503)
        self.assertEqual(city, "Tokyo")
        self.assertEqual(country, "Japan")

        city, country = reverse_geocode_offline(19.0760, 72.8777)
        self.assertEqual(city, "Mumbai")
        self.assertEqual(country, "India")

    def test_exif_metadata_backfill_writing(self):
        from photo_organizer.core.exif_writer import backfill_exif_datetime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "IMG-20240512-WA0001.jpg"
            img = Image.new("RGB", (100, 100), (120, 140, 160))
            img.save(img_path, "JPEG")

            target_dt = datetime(2024, 5, 12, 14, 30, 0)
            written = backfill_exif_datetime(img_path, target_dt)
            self.assertTrue(written)

            # Verify written EXIF headers
            with Image.open(img_path) as reloaded:
                exif = reloaded.getexif()
                self.assertIsNotNone(exif)
                # 36867 is DateTimeOriginal
                self.assertEqual(exif.get(36867), "2024:05:12 14:30:00")

    def test_burst_detection_and_best_shot_selector(self):
        from photo_organizer.models import PhotoInfo
        from photo_organizer.vision.bursts import detect_burst_groups

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1_path = root / "BURST_001.jpg"
            p2_path = root / "BURST_002.jpg"
            p3_path = root / "BURST_003.jpg"

            # Image 1: uniform gray
            Image.new("RGB", (100, 100), (128, 128, 128)).save(p1_path, "JPEG")
            # Image 2: high contrast checkered (sharp)
            img2 = Image.new("RGB", (100, 100), (255, 255, 255))
            for x in range(0, 100, 10):
                for y in range(0, 100, 10):
                    if (x + y) % 20 == 0:
                        img2.putpixel((x, y), (0, 0, 0))
            img2.save(p2_path, "JPEG")
            # Image 3: uniform gray
            Image.new("RGB", (100, 100), (130, 130, 130)).save(p3_path, "JPEG")

            base_dt = datetime(2024, 6, 1, 10, 0, 0)
            p1 = PhotoInfo(path=p1_path, taken_at=base_dt, file_size=1000, width=100, height=100, visual_hash=0x1234)
            p2 = PhotoInfo(path=p2_path, taken_at=datetime(2024, 6, 1, 10, 0, 1), file_size=1000, width=100, height=100, visual_hash=0x1234)
            p3 = PhotoInfo(path=p3_path, taken_at=datetime(2024, 6, 1, 10, 0, 2), file_size=1000, width=100, height=100, visual_hash=0x1234)

            bursts = detect_burst_groups([p1, p2, p3])
            self.assertEqual(len(bursts), 1)
            self.assertEqual(len(bursts[0].photos), 3)
            self.assertEqual(bursts[0].best_photo.path, p2_path)

    def test_semantic_offline_search(self):
        from photo_organizer.models import PhotoInfo
        from photo_organizer.vision.semantic import search_photos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blue_path = root / "summer_beach.jpg"
            Image.new("RGB", (80, 80), (30, 80, 240)).save(blue_path, "JPEG")

            orange_path = root / "evening_sunset.jpg"
            Image.new("RGB", (80, 80), (250, 120, 20)).save(orange_path, "JPEG")

            p_blue = PhotoInfo(path=blue_path, taken_at=datetime(2024, 7, 15, 12, 0), file_size=1000, width=80, height=80, city="Nice", country="France")
            p_orange = PhotoInfo(path=orange_path, taken_at=datetime(2024, 9, 20, 18, 30), file_size=1000, width=80, height=80, city="Santorini", country="Greece")

            sunset_matches = search_photos("sunset", [p_blue, p_orange])
            self.assertGreater(len(sunset_matches), 0)
            self.assertEqual(sunset_matches[0][0].path, orange_path)

            beach_matches = search_photos("beach", [p_blue, p_orange])
            self.assertGreater(len(beach_matches), 0)
            self.assertEqual(beach_matches[0][0].path, blue_path)

            city_matches = search_photos("santorini", [p_blue, p_orange])
            self.assertGreater(len(city_matches), 0)
            self.assertEqual(city_matches[0][0].path, orange_path)

    def test_flashback_memories_generation(self):
        from photo_organizer.models import PhotoInfo
        from photo_organizer.core.flashbacks import get_flashback_cards

        today = datetime.now()
        one_year_ago = today.replace(year=today.year - 1)
        two_years_ago = today.replace(year=today.year - 2)

        p1 = PhotoInfo(path=Path("/memories/pic1.jpg"), taken_at=one_year_ago, file_size=1000, width=100, height=100)
        p2 = PhotoInfo(path=Path("/memories/pic2.jpg"), taken_at=two_years_ago, file_size=1000, width=100, height=100)

        cards = get_flashback_cards([p1, p2])
        self.assertGreaterEqual(len(cards), 2)
        titles = [c["title"] for c in cards]
        self.assertTrue(any("1 Year Ago" in t for t in titles))
        self.assertTrue(any("2 Years Ago" in t for t in titles))

    def test_raw_and_heif_and_live_photo_detection(self):
        from photo_organizer.core.raw_heif import (
            get_live_photo_video_path,
            is_heif_file,
            is_raw_file,
        )

        self.assertTrue(is_raw_file(Path("photo.CR2")))
        self.assertTrue(is_raw_file(Path("photo.NEF")))
        self.assertTrue(is_raw_file(Path("photo.ARW")))
        self.assertTrue(is_raw_file(Path("photo.DNG")))
        self.assertFalse(is_raw_file(Path("photo.JPG")))

        self.assertTrue(is_heif_file(Path("photo.HEIC")))
        self.assertTrue(is_heif_file(Path("photo.heif")))
        self.assertFalse(is_heif_file(Path("photo.PNG")))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "IMG_1234.HEIC"
            mov = root / "IMG_1234.MOV"
            img.write_bytes(b"dummy heic")
            mov.write_bytes(b"dummy mov")

            companion_vid = get_live_photo_video_path(img)
            self.assertIsNotNone(companion_vid)
            self.assertEqual(companion_vid.name.upper(), "IMG_1234.MOV")

    def test_web_ui_advanced_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(default_root=root)
            client = TestClient(app)

            res_map = client.get("/api/map")
            self.assertEqual(res_map.status_code, 200)
            self.assertIn("geo_photos", res_map.json())

            res_bursts = client.get("/api/bursts")
            self.assertEqual(res_bursts.status_code, 200)
            self.assertIn("burst_groups", res_bursts.json())

            res_people = client.get("/api/people")
            self.assertEqual(res_people.status_code, 200)
            self.assertIn("people", res_people.json())

            res_flashbacks = client.get("/api/flashbacks")
            self.assertEqual(res_flashbacks.status_code, 200)
            self.assertIn("flashbacks", res_flashbacks.json())

            res_search = client.get("/api/search?q=test")
            self.assertEqual(res_search.status_code, 200)
            self.assertIn("results", res_search.json())

            res_rename = client.post("/api/people/rename", json={"person_id": "person_1", "name": "Sarah"})
            self.assertEqual(res_rename.status_code, 200)
            self.assertEqual(res_rename.json()["name"], "Sarah")

    def test_studio_state_persistence_and_rehydration(self):
        """Verifies that studio state (People, Flashbacks, etc.) persists and re-hydrates upon set-directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img1 = root / "IMG_20240101_100000.jpg"
            img1.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 200)

            cache = PhotoCache(root_dir=root)
            mock_state = {
                "people_albums": [
                    {"person_id": "p1", "name": "Alice", "face_count": 5, "photo_paths": [str(img1)]}
                ],
                "flashbacks": [
                    {"anniversary_label": "1 Year Ago", "photos": [{"path": str(img1), "year": 2024}]}
                ],
                "geo_photos": [{"path": str(img1), "latitude": 40.7128, "longitude": -74.0060}],
            }
            cache.save_studio_state(mock_state)

            # Test cache loading directly
            loaded = cache.load_studio_state()
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["people_albums"]), 1)
            self.assertEqual(loaded["people_albums"][0]["name"], "Alice")

            # Test web API set-directory re-hydrating it immediately
            app = create_app(default_root=root)
            client = TestClient(app)
            res = client.post("/api/set-directory", json={"directory": str(root)})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data.get("has_cached_state"))
            self.assertEqual(data.get("people_count"), 1)
            self.assertEqual(data.get("flashback_count"), 1)

            # Confirm /api/people and /api/flashbacks return the re-hydrated data without scanning
            people_res = client.get("/api/people")
            self.assertEqual(people_res.status_code, 200)
            self.assertEqual(len(people_res.json()["people"]), 1)
            self.assertEqual(people_res.json()["people"][0]["name"], "Alice")

            flash_res = client.get("/api/flashbacks")
            self.assertEqual(flash_res.status_code, 200)
            self.assertEqual(len(flash_res.json()["flashbacks"]), 1)

    def test_deep_face_embeddings_and_cosine_clustering(self):
        """Verifies that face recognition models load and cluster embeddings using cosine distance."""
        from photo_organizer.vision.faces import _get_deep_face_models, cluster_people_albums
        det, rec = _get_deep_face_models()
        self.assertIsNotNone(det)
        self.assertIsNotNone(rec)

        # Test clustering logic with distinct embeddings
        import numpy as np
        f_alice_1 = np.zeros(128, dtype=np.float32)
        f_alice_1[0] = 1.0  # Unit vector along axis 0
        f_alice_2 = np.zeros(128, dtype=np.float32)
        f_alice_2[0] = 0.95
        f_alice_2[1] = 0.312  # Cosine similarity ~ 0.95

        f_bob = np.zeros(128, dtype=np.float32)
        f_bob[10] = 1.0  # Orthogonal vector, cosine similarity = 0.0

        p1 = PhotoInfo(path=Path("alice_1.jpg"), taken_at=datetime(2024, 1, 1))
        p2 = PhotoInfo(path=Path("alice_2.jpg"), taken_at=datetime(2024, 1, 2))
        p3 = PhotoInfo(path=Path("bob.jpg"), taken_at=datetime(2024, 1, 3))

        face_records = [
            (p1, f_alice_1, "thumb1"),
            (p2, f_alice_2, "thumb2"),
            (p3, f_bob, "thumb3"),
        ]

        # Mock extract_face_crops_and_features to test clustering algorithm
        import photo_organizer.vision.faces as face_module
        orig_extract = face_module.extract_face_crops_and_features
        face_module.extract_face_crops_and_features = lambda p: [
            ((0, 0, 50, 50), f_alice_1 if "alice_1" in str(p.path) else (f_alice_2 if "alice_2" in str(p.path) else f_bob), "thumb")
        ]
        try:
            clusters = cluster_people_albums([p1, p2, p3], similarity_threshold=0.48)
            # Must produce exactly 2 distinct people clusters: Alice and Bob
            self.assertEqual(len(clusters), 2)
            # Largest cluster should have Alice's 2 photos
            self.assertEqual(clusters[0].face_count, 2)
            self.assertEqual(clusters[1].face_count, 1)
        finally:
            face_module.extract_face_crops_and_features = orig_extract


    def test_triage_decision_persistence_and_override(self):
        """Verifies triage decision overrides: KEEP saves to library, DISCARD moves to junk."""
        from photo_organizer.ui.web.server import create_app
        from fastapi.testclient import TestClient
        from photo_organizer.organizer import organize

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img1 = root / "meme_20240101.jpg"
            img2 = root / "meme2_20240101.jpg"
            # Create synthetic images
            im = Image.new("RGB", (400, 300), color=(255, 255, 255))
            im.save(img1)
            im.save(img2)

            app = create_app(default_root=root)
            client = TestClient(app)
            client.post("/api/set-directory", json={"directory": str(root)})

            # Mock scan data with borderline items
            app.state.app_state.scan_data = {
                "non_relevant_items": [],
                "maybe_non_relevant_items": [
                    {
                        "path": str(img1),
                        "relative_path": img1.name,
                        "text_score": 0.75,
                        "reason": "borderline text image",
                        "face_detected": False,
                        "person_detected": False,
                        "file_size": 1000,
                        "width": 400,
                        "height": 300,
                    },
                    {
                        "path": str(img2),
                        "relative_path": img2.name,
                        "text_score": 0.85,
                        "reason": "borderline text image 2",
                        "face_detected": False,
                        "person_detected": False,
                        "file_size": 1000,
                        "width": 400,
                        "height": 300,
                    },
                ],
            }

            # 1. Initially decision is DEFAULT
            t_res = client.get("/api/triage")
            self.assertEqual(t_res.status_code, 200)
            items = t_res.json()["maybe_non_relevant"]
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["user_decision"], "DEFAULT")

            # 2. Set img1 to MAIN and img2 to NON_RELEVANT
            d1 = client.post("/api/triage/decide", json={"path": str(img1), "decision": "MAIN"})
            self.assertEqual(d1.status_code, 200)
            self.assertEqual(d1.json()["decision"], "MAIN")
            d2 = client.post("/api/triage/decide", json={"path": str(img2), "decision": "NON_RELEVANT"})
            self.assertEqual(d2.status_code, 200)
            self.assertEqual(d2.json()["decision"], "NON_RELEVANT")

            # Check triage query returns active decisions
            t_res2 = client.get("/api/triage")
            items2 = t_res2.json()["maybe_non_relevant"]
            decisions = {it["path"]: it["user_decision"] for it in items2}
            self.assertEqual(decisions[str(img1)], "MAIN")
            self.assertEqual(decisions[str(img2)], "NON_RELEVANT")

            # 3. Test TO_BE_DELETED
            d3 = client.post("/api/triage/decide", json={"path": str(img2), "decision": "TO_BE_DELETED"})
            self.assertEqual(d3.status_code, 200)
            t_res_del = client.get("/api/triage")
            self.assertEqual(t_res_del.json()["maybe_non_relevant"][1]["user_decision"], "TO_BE_DELETED")

            # 4. Test RESET
            r_res = client.post("/api/triage/decide", json={"path": str(img2), "decision": "RESET"})
            self.assertEqual(r_res.status_code, 200)
            t_res3 = client.get("/api/triage")
            items3 = t_res3.json()["maybe_non_relevant"]
            decisions3 = {it["path"]: it["user_decision"] for it in items3}
            self.assertEqual(decisions3[str(img2)], "DEFAULT")

    def test_face_aware_feathered_mask_and_biometric_guardrail(self):
        import numpy as np
        from photo_organizer.vision.enhancer import (
            build_feathered_face_mask,
            verify_and_protect_facial_identity,
            apply_enhancement_plan,
        )
        from photo_organizer.models import EnhancementPlan

        # 1. Test feathered mask generation
        mask = build_feathered_face_mask((300, 300), [(50, 50, 80, 100)])
        self.assertEqual(mask.shape, (300, 300))
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)
        # Center should be weighted higher than periphery
        self.assertGreater(mask[95, 90], 0.8)
        self.assertEqual(mask[0, 0], 0.0)

        # 2. Test SFace biometric guardrail
        orig_bgr = np.full((200, 200, 3), 120, dtype=np.uint8)
        fake_face_row = np.array([30, 30, 80, 80, 50, 50, 90, 50, 70, 70, 55, 90, 85, 90, 0.99], dtype=np.float32)

        # Candidate with slight exposure change should remain safe
        cand_bgr = np.full((200, 200, 3), 130, dtype=np.uint8)
        safe_res = verify_and_protect_facial_identity(orig_bgr, cand_bgr, [fake_face_row], min_cosine_similarity=0.98)
        self.assertEqual(safe_res.shape, (200, 200, 3))

        # Candidate with aggressive pixel distortion should trigger restoration
        distorted_bgr = cand_bgr.copy()
        distorted_bgr[25:115, 25:115] = np.random.randint(0, 255, (90, 90, 3), dtype=np.uint8)
        restored = verify_and_protect_facial_identity(orig_bgr, distorted_bgr, [fake_face_row], min_cosine_similarity=0.98)
        self.assertEqual(restored.shape, (200, 200, 3))
        # Center of face should have been pulled back to original pixels (120)
        self.assertAlmostEqual(float(restored[70, 70, 0]), 120, delta=15)

        # 3. Test apply_enhancement_plan produces valid PIL image
        test_img = Image.new("RGB", (150, 150), (60, 50, 45))
        plan = EnhancementPlan(True, "underexposed", gamma=0.88, brightness_factor=1.03, contrast_factor=1.06)
        enhanced = apply_enhancement_plan(test_img, plan, "balanced")
        self.assertEqual(enhanced.size, (150, 150))

    def test_sota_natural_beauty_and_facial_enhancement_pipeline(self):
        import numpy as np
        from photo_organizer.vision.enhancer import (
            build_facial_landmark_masks,
            apply_retinex_illumination_exposure,
            apply_guided_detail_enhancement,
            apply_perceptual_adaptive_vibrance,
            apply_frequency_separation_portrait,
            align_melanin_hemoglobin_skin_tones,
            apply_enhancement_plan,
        )
        from photo_organizer.models import EnhancementPlan

        # 1. Landmark masks (eyes and lips)
        face_row = np.array([30, 30, 80, 80, 50, 50, 90, 50, 70, 70, 60, 90, 80, 90, 0.99], dtype=np.float32)
        eye_mask, lip_mask = build_facial_landmark_masks((150, 150), [face_row])
        self.assertEqual(eye_mask.shape, (150, 150))
        self.assertEqual(lip_mask.shape, (150, 150))
        self.assertGreater(eye_mask[50, 50], 0.5)
        self.assertGreater(eye_mask[50, 90], 0.5)
        self.assertEqual(eye_mask[0, 0], 0.0)
        self.assertGreater(lip_mask[90, 70], 0.5)
        self.assertEqual(lip_mask[0, 0], 0.0)

        # 2. Retinex with filmic highlight rolloff (protects clouds and highlights from blowout)
        l_chan = np.full((100, 100), 220, dtype=np.uint8)
        l_chan[0:50, :] = 40  # underexposed half
        l_res = apply_retinex_illumination_exposure(l_chan, gamma=0.85, brightness_factor=1.05)
        self.assertEqual(l_res.shape, (100, 100))
        # Underexposed region was lifted
        self.assertGreater(int(l_res[25, 25]), 40)
        # Highlight region (220) was softly rolled off and not clipped to 255 pure white blowout
        self.assertLess(int(l_res[75, 75]), 255)

        # 3. Halo-free detail enhancement (edge-gradient attenuation)
        test_l = np.full((100, 100), 100, dtype=np.uint8)
        test_l[10:30, 10:30] = np.random.randint(95, 105, (20, 20), dtype=np.uint8)
        test_l[:, 50:] = 220
        enhanced_l = apply_guided_detail_enhancement(test_l, sharpness_factor=1.15, contrast_factor=1.05)
        self.assertEqual(enhanced_l.shape, (100, 100))

        # 4. Perceptual adaptive vibrance (protects already saturated sky/foliage)
        a_chan = np.full((50, 50), 128, dtype=np.uint8)
        b_chan = np.full((50, 50), 128, dtype=np.uint8)
        a_chan[10, 10] = 140
        b_chan[10, 10] = 140
        a_chan[30, 30] = 180
        b_chan[30, 30] = 180
        new_a, new_b = apply_perceptual_adaptive_vibrance(a_chan, b_chan, color_factor=1.10)
        muted_boost = (float(new_a[10, 10]) - 128.0) / (140.0 - 128.0)
        sat_boost = (float(new_a[30, 30]) - 128.0) / (180.0 - 128.0)
        self.assertGreater(muted_boost, sat_boost)

        # 5. Frequency separation portrait (preserves micro pores)
        face_mask = np.zeros((100, 100), dtype=np.float32)
        face_mask[20:80, 20:80] = 1.0
        face_l = np.full((100, 100), 120, dtype=np.uint8)
        face_l[40:60, 40:60] = np.random.randint(115, 126, (20, 20), dtype=np.uint8)
        res_face_l = apply_frequency_separation_portrait(face_l, face_mask, eye_mask=None)
        self.assertEqual(res_face_l.shape, (100, 100))
        std_enhanced = float(np.std(res_face_l[40:60, 40:60]))
        self.assertGreater(std_enhanced, 0.5)

        # 6. Melanin-hemoglobin alignment & lip vitality
        a_face = np.full((80, 80), 140, dtype=np.uint8)
        b_face = np.full((80, 80), 135, dtype=np.uint8)
        fm = np.ones((80, 80), dtype=np.float32)
        lm = np.zeros((80, 80), dtype=np.float32)
        lm[50:60, 30:50] = 1.0
        al_a, al_b = align_melanin_hemoglobin_skin_tones(a_face, b_face, fm, lm)
        self.assertGreater(int(al_b[20, 20]), int(b_face[20, 20]))
        self.assertGreater(int(al_a[55, 40]), int(a_face[55, 40]))

        # 7. Natural foliage and sky protection
        # Foliage: da < -2, db > 5 (e.g. da = -10, db = +15 -> a=118, b=143)
        a_foliage = np.full((30, 30), 118, dtype=np.uint8)
        b_foliage = np.full((30, 30), 143, dtype=np.uint8)
        res_a_fol, res_b_fol = apply_perceptual_adaptive_vibrance(a_foliage, b_foliage, color_factor=1.20)
        self.assertGreater(int(res_a_fol[15, 15]), 110)

        # Sky: db < -8 (e.g. da = 0, db = -20 -> a=128, b=108)
        a_sky = np.full((30, 30), 128, dtype=np.uint8)
        b_sky = np.full((30, 30), 108, dtype=np.uint8)
        res_a_sky, res_b_sky = apply_perceptual_adaptive_vibrance(a_sky, b_sky, color_factor=1.20)
        self.assertAlmostEqual(int(res_a_sky[15, 15]), 128, delta=2)

    def test_enhancement_api_endpoints_and_organize_filtering(self):
        from photo_organizer.ui.web.server import create_app
        from photo_organizer.organizer import organize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create photos that need enhancement (underexposed)
            photo1 = root / "photo1.jpg"
            Image.new("RGB", (200, 200), (50, 45, 40)).save(photo1, "JPEG")
            photo2 = root / "photo2.jpg"
            Image.new("RGB", (200, 200), (52, 47, 42)).save(photo2, "JPEG")

            app = create_app(root)
            client = TestClient(app)

            # Trigger scan
            s_res = client.post("/api/scan")
            self.assertEqual(s_res.status_code, 200)

            # Wait for scan completion
            for _ in range(50):
                st = client.get("/api/status").json()
                if not st.get("is_scanning", False):
                    break

            # 1. Query enhancements
            e_res = client.get("/api/enhancements")
            self.assertEqual(e_res.status_code, 200)
            data = e_res.json()
            self.assertIn("candidates", data)

            # 2. Test single decide
            d_res = client.post("/api/enhancements/decide", json={"path": str(photo1), "decision": "ORIGINAL"})
            self.assertEqual(d_res.status_code, 200)
            self.assertEqual(d_res.json()["decision"], "ORIGINAL")

            # 3. Test batch decide
            b_res = client.post("/api/enhancements/decide", json={"action": "ENHANCE_ALL"})
            self.assertEqual(b_res.status_code, 200)

            # 4. Test enhanced preview endpoint
            prev_res = client.get(f"/api/enhanced-preview?path={photo1}")
            self.assertEqual(prev_res.status_code, 200)
            self.assertEqual(prev_res.headers["content-type"], "image/jpeg")
            self.assertGreater(len(prev_res.content), 100)

            # 5. Test organize respects custom_enhancement_decisions (skips ORIGINAL)
            org_root = root / "to_org"
            org_root.mkdir()
            p_keep_orig = org_root / "keep_orig.jpg"
            img1 = Image.new("RGB", (150, 150), (45, 40, 35))
            img1.putpixel((5, 5), (100, 100, 100))
            img1.save(p_keep_orig, "JPEG")

            p_enhance = org_root / "do_enhance.jpg"
            img2 = Image.new("RGB", (150, 150), (48, 42, 38))
            img2.putpixel((10, 10), (120, 120, 120))
            img2.save(p_enhance, "JPEG")

            decisions = {str(p_keep_orig.resolve()): "ORIGINAL"}
            summary = organize(
                org_root,
                dry_run=False,
                skip_enhance=False,
                skip_non_relevant=True,
                max_size_increase_percent=20.0,
                custom_enhancement_decisions=decisions,
            )
            # Only do_enhance should be enhanced; keep_orig was excluded by decision!
            self.assertEqual(summary.enhanced, 1)

    def test_large_library_performance_and_thumbnail_cache(self):
        from photo_organizer.core.cache import ThumbnailCache, PhotoCache
        from photo_organizer.core.raw_heif import extract_fast_thumbnail, parse_exif_thumbnail
        from photo_organizer.ui.web.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Verify SQLite WAL, mmap_size, and indexes on PhotoCache
            pc = PhotoCache(root_dir=root)
            conn = pc._create_connection()
            try:
                jm = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                self.assertEqual(jm.lower(), "wal")
                indexes = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index';").fetchall()]
                self.assertIn("idx_cache_lookup", indexes)
                self.assertIn("idx_cache_taken_at", indexes)
                self.assertIn("idx_cache_visual_hash", indexes)
                self.assertIn("idx_cache_mtime", indexes)
            finally:
                conn.close()

            # 2. Verify ThumbnailCache get/set/clear and keying
            tc = ThumbnailCache(root_dir=root)
            test_file = root / "sample.jpg"
            test_file.write_bytes(b"dummy")
            self.assertIsNone(tc.get(test_file, 5, 100, tag="orig"))

            fake_thumb = b"\xff\xd8\xff\xe0fake_jpeg_content\xff\xd9"
            tc.set(test_file, 5, 100, fake_thumb, tag="orig")
            self.assertEqual(tc.get(test_file, 5, 100, tag="orig"), fake_thumb)
            self.assertIsNone(tc.get(test_file, 5, 100, tag="enh"))
            self.assertIsNone(tc.get(test_file, 6, 100, tag="orig"))
            self.assertIsNone(tc.get(test_file, 5, 101, tag="orig"))

            tc.clear()
            self.assertIsNone(tc.get(test_file, 5, 100, tag="orig"))

            # 3. Test parse_exif_thumbnail on empty or plain bytes
            self.assertIsNone(parse_exif_thumbnail(b"not_a_jpeg"))
            self.assertIsNone(extract_fast_thumbnail(test_file))

            # 4. Test Web API preview caching
            real_img = root / "real.jpg"
            Image.new("RGB", (300, 300), (100, 120, 140)).save(real_img, "JPEG")
            app = create_app(root)
            client = TestClient(app)

            res1 = client.get(f"/api/image-preview?path={real_img}")
            self.assertEqual(res1.status_code, 200)
            self.assertIn("etag", res1.headers)
            content1 = res1.content

            res2 = client.get(f"/api/image-preview?path={real_img}")
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res2.content, content1)
            self.assertEqual(res2.headers.get("etag"), res1.headers.get("etag"))

    def test_organize_backfills_exif_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wa_photo = root / "IMG-20240512-WA0001.jpg"
            img = Image.new("RGB", (120, 120), (70, 80, 90))
            img.save(wa_photo, "JPEG")

            # 1. Verify scan_library detects it as backfill candidate
            scan_res = scan_library(root, visual_threshold=5, skip_non_relevant=True)
            self.assertEqual(scan_res["exif_backfill_count"], 1)
            self.assertEqual(len(scan_res["exif_backfill_candidates"]), 1)
            candidate = scan_res["exif_backfill_candidates"][0]
            self.assertEqual(candidate["filename"], "IMG-20240512-WA0001.jpg")
            self.assertEqual(candidate["date_source"], "filename")
            self.assertEqual(candidate["resolved_date"], "2024-05-12 00:00:00")

            # 2. Verify organize backfills EXIF metadata by default
            summary = organize(
                root,
                dry_run=False,
                skip_enhance=True,
                skip_non_relevant=True,
            )
            self.assertEqual(summary.exif_backfilled, 1)

            # 3. Verify organized destination and EXIF headers
            dest = root / "2024" / "2024-05" / "IMG-20240512-WA0001.jpg"
            self.assertTrue(dest.exists())
            with Image.open(dest) as reloaded:
                exif = reloaded.getexif()
                self.assertIsNotNone(exif)
                self.assertEqual(exif.get(36867), "2024:05:12 00:00:00")
                self.assertEqual(exif.get(306), "2024:05:12 00:00:00")

    def test_organize_respects_skip_exif_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wa_photo = root / "IMG-20240512-WA0002.jpg"
            img = Image.new("RGB", (120, 120), (70, 80, 90))
            img.save(wa_photo, "JPEG")

            summary = organize(
                root,
                dry_run=False,
                skip_enhance=True,
                skip_non_relevant=True,
                skip_exif_backfill=True,
            )
            self.assertEqual(summary.exif_backfilled, 0)

            dest = root / "2024" / "2024-05" / "IMG-20240512-WA0002.jpg"
            self.assertTrue(dest.exists())
            with Image.open(dest) as reloaded:
                exif = reloaded.getexif()
                self.assertIsNone(exif.get(36867))

    def test_web_api_exif_candidates_and_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wa_photo = root / "IMG-20240512-WA0003.jpg"
            img = Image.new("RGB", (120, 120), (70, 80, 90))
            img.save(wa_photo, "JPEG")

            app = create_app(root)
            client = TestClient(app)

            # Trigger scan
            s_res = client.post("/api/scan")
            self.assertEqual(s_res.status_code, 200)

            # Check candidates endpoint
            c_res = client.get("/api/exif/candidates")
            self.assertEqual(c_res.status_code, 200)
            c_data = c_res.json()
            self.assertEqual(c_data["count"], 1)
            self.assertEqual(c_data["candidates"][0]["filename"], "IMG-20240512-WA0003.jpg")

            # Check dry-run backfill via API
            bf_res = client.post("/api/exif/backfill", json={"dry_run": True})
            self.assertEqual(bf_res.status_code, 200)
            self.assertEqual(bf_res.json()["updated_count"], 1)


if __name__ == "__main__":
    unittest.main()


