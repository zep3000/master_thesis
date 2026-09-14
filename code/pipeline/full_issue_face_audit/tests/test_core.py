from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
TEST_TMP = HERE / "local" / "test_tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)

from core import (  # noqa: E402
    assign_detector_box,
    collection_sha256,
    iou,
    make_face_composite,
    maximum_cardinality_matches,
    select_stratified_pilot,
)


class CoreTests(unittest.TestCase):
    def test_collection_hash_includes_sorted_filenames(self):
        second = TEST_TMP / "hash_b.csv"
        first = TEST_TMP / "hash_a.csv"
        second.write_bytes(b"two")
        first.write_bytes(b"one")
        try:
            self.assertEqual(
                collection_sha256([second, first]),
                collection_sha256([first, second]),
            )
        finally:
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)

    def test_two_page_detector_assignment_and_transform(self):
        page, box, side = assign_detector_box(
            "1991-0928-0140,0141_000_100__0_0_4490_3116.jpg",
            (0.60, 0.20, 0.70, 0.30),
        )
        self.assertEqual(page, "1991-0928-0141")
        self.assertEqual(side, "two_page_right")
        self.assertAlmostEqual(box[0], 0.20)
        self.assertAlmostEqual(box[2], 0.40)

    def test_maximum_cardinality_matching_is_one_to_one(self):
        faces = [
            {"face_id": "a", "bbox_rel": [0.0, 0.0, 0.2, 0.2]},
            {"face_id": "b", "bbox_rel": [0.2, 0.0, 0.4, 0.2]},
        ]
        detections = [
            {"bbox_rel": [0.0, 0.0, 0.21, 0.2]},
            {"bbox_rel": [0.19, 0.0, 0.4, 0.2]},
        ]
        matches = maximum_cardinality_matches(faces, detections, 0.1)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({item[0] for item in matches}), 2)
        self.assertEqual(len({item[1] for item in matches}), 2)

    def test_iou(self):
        self.assertEqual(iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(iou([0, 0, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]), 0.0)

    def test_stratified_pilot_is_deterministic_and_covers_cells(self):
        faces = []
        for decade in range(1940, 2010, 10):
            for status in ["found", "missed"]:
                for analyzable in [False, True]:
                    for index in range(2):
                        faces.append({
                            "face_id": f"{decade}-{status}-{analyzable}-{index}",
                            "decade": decade,
                            "detection_status": status,
                            "analyzable": analyzable,
                        })
        first = select_stratified_pilot(faces, 40, "seed")
        second = select_stratified_pilot(faces, 40, "seed")
        self.assertEqual(first, second)
        selected = {face["face_id"]: face for face in faces}
        cells = {
            (
                selected[face_id]["decade"],
                selected[face_id]["detection_status"],
                selected[face_id]["analyzable"],
            )
            for face_id in first
        }
        self.assertEqual(len(cells), 28)

    def test_composite_dimensions(self):
        path = TEST_TMP / "composite_page.jpg"
        Image.new("RGB", (800, 1200), "white").save(path)
        try:
            composite = make_face_composite(path, [200, 200, 300, 300])
            self.assertEqual(composite.size, (1536, 554))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
