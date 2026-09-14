import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_brand_industry_verification import summarize


def record(position, *, identification_wrong=False, category="correct", brand="correct"):
    return {
        "status": "complete",
        "item_position": position,
        "identification_wrong": identification_wrong,
        "category_review": None if identification_wrong else category,
        "brand_name_review": None if identification_wrong else brand,
        "corrected_brand_category": "Food" if category == "incorrect" else None,
        "corrected_brand_name": "Verified" if brand == "incorrect" else None,
        "original": {"brand_category": "Retail", "brand_name": "Model"},
    }


class SummaryTest(unittest.TestCase):
    def test_counts_scope_and_disclosure_safe_errors(self):
        records = [
            record(0),
            record(1, identification_wrong=True),
            record(2, category="incorrect"),
            record(3, brand="incorrect"),
            record(4),  # completed but beyond the declared four-record scope
        ]

        result = summarize(records, limit=4)

        self.assertEqual(result["scope"]["scorable_records"], 3)
        self.assertEqual(result["scope"]["excluded_completed_records_beyond_scope"], 1)
        self.assertEqual(result["metrics"][1]["successes"], 2)
        self.assertEqual(result["metrics"][2]["successes"], 2)
        self.assertEqual(result["metrics"][3]["successes"], 1)
        self.assertEqual(result["industry_errors"][0]["verified_category"], "Food")
        self.assertEqual(result["brand_errors"][0]["verified_name"], "Verified")
        forbidden = {"item_id", "image_id", "session_id", "created_at", "notes", "bbox_1000"}
        self.assertTrue(forbidden.isdisjoint(result["industry_errors"][0]))
        self.assertTrue(forbidden.isdisjoint(result["brand_errors"][0]))


if __name__ == "__main__":
    unittest.main()
