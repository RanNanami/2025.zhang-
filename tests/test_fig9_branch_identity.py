import csv
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.analyze_fig9_branch_identity import (
    _branch_rows,
    _lineage_rows,
    _segment_growth,
)


class Fig9BranchIdentityTests(unittest.TestCase):
    def test_branch_rows_ignore_anchor_rows_from_legacy_registry(self):
        rows = [
            {
                "actual_branch_provenance_id": "b1",
                "root_actual_anchor_id": "a1",
                "branch_depth": "0",
                "parent_branch_ids": "",
            },
            {
                "actual_branch_provenance_id": "b1",
                "root_actual_anchor_id": "",
                "branch_depth": "",
                "parent_branch_ids": "",
            },
        ]
        self.assertEqual(_branch_rows(rows), [rows[0]])

    def test_lineage_does_not_claim_parent_without_trace_evidence(self):
        rows = _lineage_rows(
            [
                {
                    "actual_record_index": "200",
                    "field": "passenger",
                    "encoded_column": "1",
                    "prematch_actual_anchor_ids": "a1|a2",
                    "selected_actual_anchor_ids": "a1",
                    "selected_segment_provenance_id": "s1",
                    "branch_continuity_status": "ACTUAL_BRANCH_MIXED_HISTORY",
                }
            ]
        )
        self.assertEqual(rows[0]["lineage_class"], "LINEAGE_UNRESOLVED")
        self.assertEqual(rows[0]["actual_lineage_id"], "")
        self.assertFalse(rows[0]["future_data_used"])

    def test_explicit_lineage_trace_is_preserved(self):
        rows = _lineage_rows(
            [],
            [
                {
                    "actual_record_index": "201",
                    "field": "passenger",
                    "encoded_column": "2",
                    "actual_anchor_id": "a2",
                    "parent_actual_anchor_ids": "a1",
                    "parent_actual_lineage_ids": "l1",
                    "actual_lineage_id": "l1",
                    "lineage_class": "LINEAGE_CONTINUATION_UNIQUE",
                    "future_data_used": "False",
                }
            ],
        )
        self.assertEqual(rows[0]["actual_lineage_id"], "l1")
        self.assertEqual(rows[0]["lineage_class"], "LINEAGE_CONTINUATION_UNIQUE")
        self.assertTrue(rows[0]["parent_lineage_available"])
        self.assertFalse(rows[0]["future_data_used"])

    def test_segment_growth_counts_only_explicit_trace_ids(self):
        rows = _segment_growth(
            [
                {
                    "segment_provenance_id": "s1",
                    "actual_anchor_ids_before": "a1|a2",
                    "actual_anchor_ids_after": "a1|a2",
                    "actual_branch_ids_before": "b1|b2",
                    "actual_branch_ids_after": "b1|b2",
                }
            ],
            [
                {
                    "selected_segment_provenance_id": "s1",
                    "reinforced": "True",
                }
            ],
        )
        self.assertEqual(rows[0]["anchors_per_segment_observed"], 2)
        self.assertEqual(rows[0]["branches_per_segment_observed"], 2)
        self.assertEqual(rows[0]["reinforcements_per_segment"], 1)
        self.assertFalse(rows[0]["lineage_identity_available"])


if __name__ == "__main__":
    unittest.main()
