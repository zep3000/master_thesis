import csv
import json
import tempfile
import unittest
from pathlib import Path

from publish_human_annotations import (
    publish_coder,
    publish_development_export,
    publish_full_issue_scans,
)


class PublicationExportTests(unittest.TestCase):
    def test_coder_export_removes_identity_and_comments(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            destination = root / "coder-a.jsonl"
            record = {
                "assignee_name": "Private Name",
                "assignment_code": "private-code",
                "annotation_set_id": "keep-this",
                "session_code": "keep-session",
                "path": f"{chr(67)}:{chr(92)}private{chr(92)}images{chr(92)}page.jpg",
                "urgent_comments": [{"text": "remove this"}],
            }
            source.write_text(
                "".join(json.dumps(record) + "\n" for _ in range(300)), encoding="utf-8"
            )

            entry = publish_coder(source, destination, "A")
            published = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(entry["records"], 300)
            self.assertEqual(published["assignee_name"], "A")
            self.assertEqual(published["annotation_set_id"], "keep-this")
            self.assertEqual(published["path"], "source-images/page.jpg")
            self.assertNotIn("assignment_code", published)
            self.assertNotIn("session_code", published)
            self.assertNotIn("urgent_comments", published)

    def test_full_issue_export_validates_expected_scope(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            remaining = 1599
            for index in range(35):
                rows = remaining // (35 - index)
                remaining -= rows
                path = source / f"ECON-2000-{index + 1:04d}_annotations.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "issue_page_identifier",
                            "x1_rel",
                            "y1_rel",
                            "x2_rel",
                            "y2_rel",
                            "analyzable",
                        ],
                    )
                    writer.writeheader()
                    for row in range(rows):
                        writer.writerow(
                            {
                                "issue_page_identifier": f"page-{index}-{row}",
                                "x1_rel": 0.1,
                                "y1_rel": 0.2,
                                "x2_rel": 0.3,
                                "y2_rel": 0.4,
                                "analyzable": "true",
                            }
                        )

            entries, scope = publish_full_issue_scans(source, destination)

            self.assertEqual(len(entries), 35)
            self.assertEqual(scope["records"], 1599)
            self.assertEqual(scope["unique_page_identifiers"], 1599)

    def test_development_export_selects_assignment_and_removes_codes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "published.json"
            source.write_text(
                json.dumps(
                    {
                        "annotation_set": {"id": "keep-set"},
                        "annotations": [
                            {
                                "image_id": "one",
                                "assignee_name": "Private Name",
                                "assignment_code": "selected",
                                "payload": {
                                    "session": {"session_code": "remove"},
                                    "copied_from": {
                                        "source_assignment_code": "remove-too",
                                        "source_assignment_id": "keep-id",
                                    },
                                    "urgent_comments": ["remove"],
                                },
                            },
                            {
                                "image_id": "two",
                                "assignee_name": "Private Name",
                                "assignment_code": "selected",
                                "payload": {},
                            },
                            {
                                "image_id": "stray",
                                "assignment_code": "other",
                                "payload": {},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            entry = publish_development_export(
                source,
                destination,
                expected_records=2,
                label="D",
                assignment_code="selected",
            )
            published = json.loads(destination.read_text(encoding="utf-8"))
            serialized = json.dumps(published)

            self.assertEqual(entry["records"], 2)
            self.assertEqual(
                entry["transformations"]["other_assignment_records_excluded"], 1
            )
            self.assertEqual(
                {row["assignee_name"] for row in published["annotations"]}, {"D"}
            )
            self.assertIn('"source_assignment_id": "keep-id"', serialized)
            self.assertNotIn("assignment_code", serialized)
            self.assertNotIn("session_code", serialized)
            self.assertNotIn("urgent_comments", serialized)


if __name__ == "__main__":
    unittest.main()
