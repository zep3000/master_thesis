from __future__ import annotations

import hashlib
import json
import shutil
import time
import unittest
from pathlib import Path

from clean_incomplete_pages import clean_run, read_jsonl


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CleanIncompletePagesTests(unittest.TestCase):
    def test_cleanup_is_verified_atomic_and_idempotent(self):
        work_root = Path(__file__).resolve().parent / ".test_work"
        root = work_root / f"cleanup-{time.time_ns()}"
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, True)
        try:
            clean_id = "example-complete-page"
            excluded = [
                "example-incomplete-assembly",
                "example-missing-structure-a",
                "example-missing-structure-b",
            ]
            (root / "manifest.json").write_text(
                json.dumps({"images": [{"image_id": value} for value in [clean_id, *excluded]]}),
                encoding="utf-8",
            )
            write_jsonl(
                root / "pages" / "structure_s2.jsonl",
                [
                    {"task_key": clean_id, "ok": True, "model_annotation": {}},
                    {"task_key": excluded[0], "ok": True, "model_annotation": {}},
                    {"task_key": excluded[1], "ok": False, "error": "failed"},
                    {"task_key": excluded[2], "ok": False, "error": "failed"},
                ],
            )
            write_jsonl(
                root / "pages" / "entities_s2_direct_gaze.jsonl",
                [
                    {"task_key": f"{clean_id}::person_1", "ok": True},
                    {"task_key": f"{excluded[0]}::person_1", "ok": False},
                ],
            )
            write_jsonl(
                root / "pages" / "assembled" / "final.jsonl",
                [
                    {"image_id": clean_id, "ok": True, "annotation": {}},
                    {"image_id": excluded[0], "ok": False, "annotation": {}},
                ],
            )
            write_jsonl(
                root / "raw_values" / "pages.jsonl",
                [
                    {"image_id": clean_id, "task_key": f"{clean_id}::person_1"},
                    {"image_id": excluded[0], "task_key": f"{excluded[0]}::person_1"},
                ],
            )
            write_jsonl(
                root / "normalization" / "pages.jsonl",
                [
                    {"task_key": f"{clean_id}::person_1", "field": "example"},
                    {"task_key": f"{excluded[0]}::person_1", "field": "example"},
                ],
            )
            ledger = root / "request_ledger.jsonl"
            write_jsonl(
                ledger,
                [
                    {"task_key": clean_id, "ok": True},
                    {"task_key": excluded[0], "ok": False},
                ],
            )
            ledger_before = digest(ledger)
            structure = root / "pages" / "structure_s2.jsonl"
            structure_before = digest(structure)

            preview = clean_run(
                root,
                apply=False,
                expected_manifest_pages=4,
                expected_baseline_pages=1,
            )
            self.assertFalse(preview["applied"])
            self.assertEqual(structure_before, digest(structure))
            self.assertFalse((root / "analysis_baseline.json").exists())

            applied = clean_run(
                root,
                apply=True,
                expected_manifest_pages=4,
                expected_baseline_pages=1,
            )
            self.assertTrue(applied["applied"])
            self.assertEqual(ledger_before, digest(ledger))
            self.assertEqual(
                [clean_id],
                [record["image_id"] for record in read_jsonl(root / "pages" / "assembled" / "final.jsonl")],
            )
            audit = root / "analysis_baseline.json"
            self.assertTrue(audit.exists())
            audit_before = digest(audit)

            repeated = clean_run(
                root,
                apply=True,
                expected_manifest_pages=4,
                expected_baseline_pages=1,
            )
            self.assertEqual(0, sum(item["records_removed"] for item in repeated["result_files"]))
            self.assertEqual(audit_before, digest(audit))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
