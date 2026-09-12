"""Offline regression tests for resumability and bounded request construction."""

from __future__ import annotations

import json
import shutil
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run
import matching
import prepare_processing_order


class FakeClient:
    calls = 0

    def create(self, content, max_tokens):
        type(self).calls += 1
        return {
            "choices": [{"message": {"content": '{"value": "ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        }


class NoAdClient:
    def create(self, content, max_tokens):
        return {
            "choices": [{"message": {"content": '{"advertisements": [], "no_qualifying_ad_reason": "no_ads_on_page"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
        }


class ProductionRunnerTests(unittest.TestCase):
    def workdir(self, label: str) -> Path:
        root = run.HERE / ".test_work" / f"{label}-{time.time_ns()}"
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def test_frozen_manifest_does_not_shift_after_source_change(self):
        root = self.workdir("manifest")
        source = root / "images"
        source.mkdir()
        (source / "2000-0001.jpg").write_bytes(b"one")
        target = root / "manifest.json"
        first = run.freeze_directory_manifest(source, target)
        (source / "1999-0001.jpg").write_bytes(b"two")
        resumed = run.freeze_directory_manifest(source, target)
        self.assertEqual(first["images"], resumed["images"])
        self.assertEqual(1, len(resumed["images"]))

    def test_decade_processing_order_preserves_prefix_and_is_complete(self):
        images = []
        for index, year in enumerate([1940, 1950, 1960, 1970, 1980, 1990, 2000, 1961, 1971]):
            images.append({
                "index": index,
                "image_id": f"{year}-{index:04d}",
                "filename": f"{year}-{index:04d}.jpg",
                "year": year,
                "source_size_bytes": index + 1,
            })
        manifest = {"images": images}
        payload = prepare_processing_order.build(manifest, preserve_prefix=2, seed=123)
        root = self.workdir("processing-order")
        path = root / "order.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        ordered, metadata = run.ordered_manifest_images(manifest, path)
        self.assertEqual([row["image_id"] for row in images[:2]], [row["image_id"] for row in ordered[:2]])
        self.assertEqual({row["image_id"] for row in images}, {row["image_id"] for row in ordered})
        self.assertEqual("decade_deficit_balanced_after_2_seed123", metadata["order_id"])

    def test_processing_order_rejects_incomplete_permutation(self):
        manifest = {"images": [
            {"image_id": "a", "filename": "a.jpg", "year": 1940, "source_size_bytes": 1},
            {"image_id": "b", "filename": "b.jpg", "year": 1950, "source_size_bytes": 1},
        ]}
        root = self.workdir("bad-processing-order")
        path = root / "order.json"
        path.write_text(json.dumps({"image_ids": ["a", "a"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete, duplicate-free permutation"):
            run.ordered_manifest_images(manifest, path)

    def test_execute_jobs_builds_content_lazily_and_skips_successes(self):
        root = self.workdir("execute")
        result = root / "results.jsonl"
        ledger = root / "ledger.jsonl"
        endpoint_output = root / "output"
        run.append_jsonl(result, {"ok": True, "task_key": "task_0", "model_annotation": {"value": "ok"}})
        built = []

        def jobs():
            for index in range(10):
                task_key = f"task_{index}"
                yield {
                    "task_key": task_key,
                    "content_factory": lambda task_key=task_key: built.append(task_key) or [{"type": "text", "text": task_key}],
                    "max_tokens": 10,
                    "normalize": lambda raw: raw,
                    "meta": {"run_id": "test", "cohort": "test", "stage": "test"},
                }

        budget = run.Budget(100, ledger)
        FakeClient.calls = 0
        with patch.object(run, "OUTPUT", endpoint_output), patch.object(run, "make_client", return_value=FakeClient()):
            summary = run.execute_jobs(jobs(), result, budget, workers=2, max_in_flight=3)
            self.assertEqual(9, FakeClient.calls)
            self.assertNotIn("task_0", built)
            self.assertEqual(3, summary["max_in_flight"])
            run.execute_jobs(jobs(), result, budget, workers=2, max_in_flight=3)
            self.assertEqual(9, FakeClient.calls)

        successful = {row["task_key"] for row in run.iter_jsonl(result) if row.get("ok")}
        self.assertEqual({f"task_{index}" for index in range(10)}, successful)
        self.assertEqual(9, sum(1 for _ in run.iter_jsonl(ledger)))

    def test_execute_mixed_jobs_keeps_stage_outputs_separate(self):
        root = self.workdir("mixed")
        ledger = root / "ledger.jsonl"
        endpoint_output = root / "output"

        def jobs(stage):
            return [{
                "task_key": f"{stage}_{index}",
                "content": [{"type": "text", "text": stage}],
                "max_tokens": 10,
                "normalize": lambda raw: raw,
                "meta": {"run_id": "test", "cohort": "test", "stage": stage},
            } for index in range(2)]

        structure_result = root / "structure.jsonl"
        entity_result = root / "entities.jsonl"
        groups = [
            {"name": "entities", "jobs": jobs("entities"), "result_path": entity_result, "workers": 2, "max_in_flight": 2},
            {"name": "structure", "jobs": jobs("structure"), "result_path": structure_result, "workers": 2, "max_in_flight": 2},
        ]
        with patch.object(run, "OUTPUT", endpoint_output), patch.object(run, "make_client", return_value=FakeClient()):
            summaries = run.execute_mixed_jobs(groups, run.Budget(20, ledger))
        self.assertEqual({"structure_0", "structure_1"}, {row["task_key"] for row in run.iter_jsonl(structure_result)})
        self.assertEqual({"entities_0", "entities_1"}, {row["task_key"] for row in run.iter_jsonl(entity_result)})
        self.assertTrue(summaries["structure"]["mixed_execution"])
        self.assertTrue(summaries["entities"]["mixed_execution"])
        self.assertEqual(4, sum(1 for _ in run.iter_jsonl(ledger)))

    def test_structure_filters_children_before_non_group_cap(self):
        people = [
            {"face_bbox_1000": [800, 100 + index, 850, 150 + index]}
            for index in range(9)
        ] + [
            {"face_bbox_1000": [100 + index * 60, 200, 140 + index * 60, 260]}
            for index in range(3)
        ]
        annotation = run.normalize_structure({"advertisements": [{
            "bbox_1000": [0, 0, 700, 700],
            "ad_category": "Retail",
            "face_depiction_count_band": "9",
            "people": people,
        }]}, "cap-test")
        retained = annotation["advertisements"][0]["people"]
        self.assertEqual(["ad_1_person_10", "ad_1_person_11", "ad_1_person_12"], [person["person_id"] for person in retained])

    def test_warc_repairs_are_bounded_and_empty_ads_do_not_fail(self):
        raw = {"advertisements": [
            {
                "bbox_1000": [0, 0, 400, 400],
                "ad_category": ["Travel & tourism"],
                "face_depiction_count_band": "1",
                "people": [{"face_bbox_1000": [100, 100, 200, 200]}],
            },
            {
                "bbox_1000": [500, 0, 900, 400],
                "ad_category": ["Education"],
                "face_depiction_count_band": "1",
                "people": [],
            },
            {
                "bbox_1000": [0, 500, 400, 900],
                "ad_category": "Unmapped sector",
                "face_depiction_count_band": "1",
                "people": [{"face_bbox_1000": [100, 600, 200, 700]}],
            },
        ]}
        annotation = run.normalize_structure(raw, "warc-test")
        self.assertEqual(2, len(annotation["advertisements"]))
        self.assertEqual("Transport & tourism", annotation["advertisements"][0]["ad_category"])
        self.assertIsNone(annotation["advertisements"][1]["ad_category"])
        self.assertIn("ad_category_unresolved", annotation["advertisements"][1]["review_flags"])

    def test_terminal_failure_classification_is_narrow(self):
        validation = {
            "ok": False,
            "http_status": 400,
            "error": "OpenRouter HTTP 400: Provider returned error",
            "error_metadata": {"raw": "Supplied image did not pass validation checks."},
            "error_category": "provider",
            "requested_max_tokens": 1300,
        }
        exhausted = {
            "ok": False,
            "error_category": "response_truncated",
            "requested_max_tokens": run.RECOVERY_MAX_TOKENS,
        }
        transient = {"ok": False, "http_status": 429, "error_category": "rate_limit"}
        self.assertEqual("provider_input_image_validation", run.terminal_failure_reason(validation))
        self.assertEqual("output_limit_exhausted", run.terminal_failure_reason(exhausted))
        self.assertIsNone(run.terminal_failure_reason(transient))

    def test_budget_exhaustion_writes_endpoint_summary_before_raising(self):
        root = self.workdir("budget-summary")
        result = root / "results.jsonl"
        ledger = root / "ledger.jsonl"
        endpoint_output = root / "output"
        jobs = [{
            "task_key": f"task_{index}",
            "content": [{"type": "text", "text": "test"}],
            "max_tokens": 10,
            "normalize": lambda raw: raw,
            "meta": {"run_id": "test", "cohort": "test", "stage": "test"},
        } for index in range(3)]
        with patch.object(run, "OUTPUT", endpoint_output), patch.object(run, "make_client", return_value=FakeClient()):
            with self.assertRaises(run.BudgetExhausted):
                run.execute_jobs(jobs, result, run.Budget(1, ledger), workers=1, max_in_flight=1)
        summary = json.loads((endpoint_output / "endpoint_behavior" / "test_test.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["execution_complete"])
        self.assertIn("BudgetExhausted", summary["fatal_error"])

    def test_token_pacer_uses_stage_size_and_single_backoff_per_429_wave(self):
        root = self.workdir("pacing")
        clock = [100.0]

        def advance(seconds):
            clock[0] += seconds

        with patch.object(run.time, "monotonic", side_effect=lambda: clock[0]), patch.object(run.time, "sleep", side_effect=advance):
            budget = run.Budget(10, root / "ledger.jsonl", target_tokens_per_minute=850_000)
            budget.wait_for_endpoint("entities")
            budget.wait_for_endpoint("entities")
            entity_spacing = clock[0] - 100.0
            self.assertAlmostEqual(1_800 * 60 / 850_000, entity_spacing, places=3)
            budget.cooldown(30)
            self.assertEqual(1.25, budget.pacing_snapshot()["rate_backoff_multiplier"])
            budget.cooldown(30)
            self.assertEqual(1.25, budget.pacing_snapshot()["rate_backoff_multiplier"])
            clock[0] += 31
            budget.cooldown(30)
            self.assertEqual(1.25, budget.pacing_snapshot()["rate_backoff_multiplier"])
            clock[0] += run.PACING_BACKOFF_QUIET_GRACE_SECONDS + run.PACING_BACKOFF_HALF_LIFE_SECONDS
            self.assertAlmostEqual(1.125, budget.pacing_snapshot()["rate_backoff_multiplier"], places=3)

    def test_optimal_matcher_maximizes_cardinality_before_overlap_score(self):
        def item(name, box):
            return matching.Item("page", "ad", name, {}, box)

        human = [item("h1", [0, 0, 10, 10]), item("h2", [4, 0, 14, 10])]
        predicted = [item("p1", [2, 0, 12, 10]), item("p2", [0, 0, 6, 10])]
        greedy = matching.pair_items(human, predicted, "people", "strict")
        optimal = matching.optimal_pair_items(human, predicted, "people", "strict")
        self.assertEqual(1, len(greedy))
        self.assertEqual(2, len(optimal))
        self.assertEqual({("h1", "p2"), ("h2", "p1")}, {(pair.human.item_id, pair.predicted.item_id) for pair in optimal})

    def test_one_command_orchestration_and_audit_without_entities(self):
        root = self.workdir("run-all")
        source = root / "images"
        source.mkdir()
        Image.new("RGB", (120, 180), "white").save(source / "2000-0101-0001.jpg", "JPEG")
        Image.new("RGB", (120, 180), "white").save(source / "2000-0101-0002.jpg", "JPEG")
        args = Namespace(
            run_name="offline_test",
            input_dir=source,
            start=0,
            limit=2,
            batch_pages=1,
            page_workers=2,
            entity_workers=2,
            page_max_in_flight=None,
            entity_max_in_flight=None,
            request_budget=10,
            recovery_passes=0,
            variant="s2",
            style="direct",
            gaze="yes",
            crop_cache="none",
            dry_run=False,
        )
        with (
            patch.object(run, "BASE_OUTPUT", root / "output"),
            patch.object(run, "OUTPUT", root / "output"),
            patch.object(run, "LEDGER", root / "output" / "ledger.jsonl"),
            patch.object(run, "ACTIVE_MANIFEST", None),
            patch.object(run, "make_client", return_value=NoAdClient()),
            patch.dict(run.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        ):
            self.assertEqual(0, run.run_all_stage(args))
            status = json.loads((root / "output" / "production" / "offline_test" / "run_status.json").read_text(encoding="utf-8"))
            self.assertTrue(status["all_selected_complete"])
            self.assertEqual(2, status["complete_assembled_pages"])

    def test_run_all_budget_exhaustion_assembles_and_audits_partial_output(self):
        root = self.workdir("run-all-budget")
        source = root / "images"
        source.mkdir()
        for index in range(2):
            Image.new("RGB", (120, 180), "white").save(source / f"2000-0101-000{index + 1}.jpg", "JPEG")
        args = Namespace(
            run_name="budget_test",
            input_dir=source,
            start=0,
            limit=2,
            batch_pages=2,
            page_workers=1,
            entity_workers=1,
            page_max_in_flight=1,
            entity_max_in_flight=1,
            request_budget=1,
            recovery_passes=0,
            variant="s2",
            style="direct",
            gaze="yes",
            crop_cache="none",
            dry_run=False,
        )
        with (
            patch.object(run, "BASE_OUTPUT", root / "output"),
            patch.object(run, "OUTPUT", root / "output"),
            patch.object(run, "LEDGER", root / "output" / "ledger.jsonl"),
            patch.object(run, "ACTIVE_MANIFEST", None),
            patch.object(run, "make_client", return_value=NoAdClient()),
            patch.dict(run.os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        ):
            self.assertEqual(2, run.run_all_stage(args))
            status_path = root / "output" / "production" / "budget_test" / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertFalse(status["all_selected_complete"])
            self.assertEqual(1, status["complete_assembled_pages"])
            self.assertIn("request budget exhausted", status["stopped_reason"])


if __name__ == "__main__":
    unittest.main()
