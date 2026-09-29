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


if __name__ == "__main__":
    unittest.main()

