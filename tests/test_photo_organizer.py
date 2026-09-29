import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from photo_organizer import (
    ImageQualityMetrics,
    NonRelevantResult,
    PhotoInfo,
    allowed_enhanced_size,
    build_enhancement_plan,
    date_from_filename,
    duplicate_paths_to_move,
    discover_images,
    enhancement_quality_steps,
    flattened_pixels,
    detect_face,
    inspect_photo,
    move_file,
    organize,
    overwrite_existing_file_from_temp,
    populate_hashes_for_exact_candidates,
    restore_access_modified_times,
)


class PhotoOrganizerTests(unittest.TestCase):
    def test_flattened_pixels_prefers_new_pillow_api(self):
        class FakeImage:
            def get_flattened_data(self):
                return [1, 2, 3]

            def getdata(self):
                raise AssertionError("old API should not be used")

        self.assertEqual(flattened_pixels(FakeImage()), [1, 2, 3])

    def test_detect_face_returns_unavailable_on_opencv_detector_error(self):
        class FakeCv2:
            COLOR_RGB2GRAY = 1

            class error(Exception):
                pass

            def cvtColor(self, _array, _code):
                return "gray"

            def flip(self, _gray, _code):
                return "flipped"

        class FakeNp:
            def array(self, _image):
                return "array"

        class FakeCascade:
            def detectMultiScale(self, *_args, **_kwargs):
                raise FakeCv2.error("detector failure")

        with patch("photo_organizer.cv2", FakeCv2()), patch(
            "photo_organizer.np", FakeNp()
        ), patch("photo_organizer.FACE_CASCADES", [FakeCascade()]), patch(
            "photo_organizer.analysis_image", return_value=object()
        ):
            self.assertEqual(detect_face(object()), (False, False))

    def test_allowed_enhanced_size_uses_percent_increase(self):
        self.assertEqual(allowed_enhanced_size(1000, 20), 1200)
        self.assertEqual(allowed_enhanced_size(1000, 0), 1000)

    def test_enhancement_quality_steps_trade_speed_for_search_depth(self):
        self.assertLess(
            len(enhancement_quality_steps("JPEG", "fast")),
            len(enhancement_quality_steps("JPEG", "best")),
        )
        self.assertEqual(enhancement_quality_steps("PNG", "balanced"), (None,))

    def test_enhancement_plan_skips_text_heavy_images(self):
        plan = build_enhancement_plan(
            ImageQualityMetrics(
                brightness=70,
                contrast=20,
                sharpness=30,
                saturation=20,
                noise=2,
                text_score=0.8,
            )
        )

        self.assertFalse(plan.should_enhance)
        self.assertIn("text-heavy", plan.reason)

    def test_enhancement_plan_targets_underexposed_low_contrast_images(self):
        plan = build_enhancement_plan(
            ImageQualityMetrics(
                brightness=65,
                contrast=24,
                sharpness=70,
                saturation=24,
                noise=3,
                text_score=0.2,
            )
        )

        self.assertTrue(plan.should_enhance)
        self.assertLess(plan.gamma, 1.0)
        self.assertGreater(plan.contrast_factor, 1.0)

    def test_date_from_filename_handles_common_camera_names(self):
        self.assertEqual(
            date_from_filename(Path("IMG_20240131_153045.jpg")),
            datetime(2024, 1, 31, 15, 30, 45),
        )
        self.assertEqual(
            date_from_filename(Path("holiday-2023-07-04.png")),
            datetime(2023, 7, 4),
        )

    def test_hashes_are_only_populated_for_same_size_exact_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unique = root / "unique_20240101.jpg"
            first = root / "first_20240101.jpg"
            second = root / "second_20240101.jpg"
            unique.write_bytes(b"unique bytes")
            first.write_bytes(b"same")
            second.write_bytes(b"same")

            photos = [inspect_photo(path) for path in (unique, first, second)]
            self.assertTrue(all(photo.sha256 is None for photo in photos))

            populate_hashes_for_exact_candidates(photos, worker_count=1)

            by_name = {photo.path.name: photo for photo in photos}
            self.assertIsNone(by_name["unique_20240101.jpg"].sha256)
            self.assertEqual(
                by_name["first_20240101.jpg"].sha256,
                by_name["second_20240101.jpg"].sha256,
            )

    def test_duplicate_selection_keeps_highest_resolution_visual_match(self):
        low = PhotoInfo(
            path=Path("low.jpg"),
            file_size=500,
            sha256="a",
            taken_at=datetime(2024, 1, 1),
            date_source="exif",
            width=800,
            height=600,
            visual_hash=123,
        )
        high = PhotoInfo(
            path=Path("high.jpg"),
            file_size=400,
            sha256="b",
            taken_at=datetime(2024, 1, 1),
            date_source="exif",
            width=1600,
            height=1200,
            visual_hash=123,
        )

        duplicates = duplicate_paths_to_move([low, high], visual_threshold=0)

        self.assertIn(low.path, duplicates)
        self.assertNotIn(high.path, duplicates)

    def test_move_file_renames_collisions_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            existing = root / "2024" / "2024-01" / "source.jpg"
            source.write_bytes(b"source")
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"existing")

            moved, final_path = move_file(source, existing, dry_run=False)

            self.assertTrue(moved)
            self.assertEqual(final_path.name, "source__1.jpg")
            self.assertEqual(final_path.read_bytes(), b"source")
            self.assertEqual(existing.read_bytes(), b"existing")

    def test_in_place_overwrite_restores_modified_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "photo.jpg"
            candidate = root / "candidate.jpg"
            original.write_bytes(b"original longer data")
            candidate.write_bytes(b"enhanced")
            original_time_ns = 1_700_000_000_123_456_789
            original_access_ns = 1_700_000_001_123_456_789
            os.utime(original, ns=(original_access_ns, original_time_ns))
            original_stat = original.stat()

            overwrite_existing_file_from_temp(original, candidate)
            restore_access_modified_times(original, original_stat)

            self.assertEqual(original.read_bytes(), b"enhanced")
            self.assertEqual(original.stat().st_mtime_ns, original_stat.st_mtime_ns)

    def test_discover_images_skips_to_be_deleted_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "IMG_20240101.jpg"
            skipped = root / "to-be-deleted" / "2024" / "2024-01" / "old.jpg"
            non_relevant = root / "non-relevant" / "2024" / "2024-01" / "text.jpg"
            maybe = root / "maybe-non-relevant" / "2024" / "2024-01" / "maybe.jpg"
            kept.write_bytes(b"kept")
            skipped.parent.mkdir(parents=True)
            skipped.write_bytes(b"skipped")
            non_relevant.parent.mkdir(parents=True)
            non_relevant.write_bytes(b"non-relevant")
            maybe.parent.mkdir(parents=True)
            maybe.write_bytes(b"maybe")

            images = discover_images(
                root,
                [
                    root / "to-be-deleted",
                    root / "non-relevant",
                    root / "maybe-non-relevant",
                ],
            )

            self.assertEqual(images, [kept])

    def test_organize_moves_non_relevant_images_to_separate_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "message_20240101.jpg"
            image.write_bytes(b"not really an image, classifier is mocked")

            result = NonRelevantResult(
                is_non_relevant=True,
                reason="text-heavy image",
                text_score=0.9,
                face_detected=False,
                face_detector_available=True,
            )
            with patch("photo_organizer.classify_non_relevant", return_value=result):
                summary = organize(
                    root=root,
                    dry_run=False,
                    visual_threshold=0,
                    to_delete_folder_name="to-be-deleted",
                    skip_enhance=True,
                )

            moved = root / "non-relevant" / "2024" / "2024-01" / image.name
            self.assertTrue(moved.exists())
            self.assertEqual(summary.non_relevant_moved, 1)
            self.assertEqual(summary.organized_moved, 0)

    def test_organize_moves_borderline_images_to_maybe_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "borderline_20240101.jpg"
            image.write_bytes(b"classifier is mocked")

            result = NonRelevantResult(
                is_non_relevant=False,
                is_maybe_non_relevant=True,
                reason="borderline text-heavy image",
                text_score=0.75,
                face_detected=False,
                face_detector_available=True,
            )
            with patch("photo_organizer.classify_non_relevant", return_value=result):
                summary = organize(
                    root=root,
                    dry_run=False,
                    visual_threshold=0,
                    to_delete_folder_name="to-be-deleted",
                    skip_enhance=True,
                )

            moved = root / "maybe-non-relevant" / "2024" / "2024-01" / image.name
            self.assertTrue(moved.exists())
            self.assertEqual(summary.maybe_non_relevant_moved, 1)
            self.assertEqual(summary.organized_moved, 0)

    def test_organize_moves_exact_duplicate_to_to_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "IMG_20240101_010101.jpg"
            second = root / "copy_20240101_010101.jpg"
            first.write_bytes(b"same bytes")
            second.write_bytes(b"same bytes")

            summary = organize(
                root=root,
                dry_run=False,
                visual_threshold=0,
                to_delete_folder_name="to-be-deleted",
                skip_enhance=True,
            )

            kept = list((root / "2024" / "2024-01").glob("*.jpg"))
            moved_duplicates = list(
                (root / "to-be-deleted" / "2024" / "2024-01").glob("*.jpg")
            )

            self.assertEqual(summary.scanned, 2)
            self.assertEqual(summary.duplicates_moved, 1)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(moved_duplicates), 1)

    def test_organize_removes_empty_source_folders_after_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_folder = root / "old" / "nested"
            source_folder.mkdir(parents=True)
            image = source_folder / "IMG_20240101.jpg"
            image.write_bytes(b"image bytes")

            summary = organize(
                root=root,
                dry_run=False,
                visual_threshold=0,
                to_delete_folder_name="to-be-deleted",
                skip_enhance=True,
                skip_non_relevant=True,
            )

            self.assertFalse(source_folder.exists())
            self.assertFalse((root / "old").exists())
            self.assertTrue((root / "2024" / "2024-01" / image.name).exists())
            self.assertEqual(summary.empty_dirs_removed, 2)

    def test_organize_keeps_source_folder_when_non_image_file_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_folder = root / "old"
            source_folder.mkdir()
            image = source_folder / "IMG_20240101.jpg"
            note = source_folder / "note.txt"
            image.write_bytes(b"image bytes")
            note.write_text("keep this folder", encoding="utf-8")

            summary = organize(
                root=root,
                dry_run=False,
                visual_threshold=0,
                to_delete_folder_name="to-be-deleted",
                skip_enhance=True,
                skip_non_relevant=True,
            )

            self.assertTrue(source_folder.exists())
            self.assertTrue(note.exists())
            self.assertEqual(summary.empty_dirs_removed, 0)


if __name__ == "__main__":
    unittest.main()
