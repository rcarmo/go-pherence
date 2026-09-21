#!/usr/bin/env python3
"""Model-free exporter policy tests; pyarrow is only imported by main()."""
import importlib.util
from pathlib import Path
import unittest
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("exporter", Path(__file__).with_name("jevlike-export-parquet.py"))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class PilotSelectionTest(unittest.TestCase):
    def test_deterministic_group_selection_and_fiction_exclusion(self):
        rows = []
        for group in range(20):
            for label in range(3):
                rows.append(dict(promptID=group, pairID=f"{group}-{label}", premise=f"p{group}", hypothesis=f"h{label}", genre="government", label=label))
        rows.append(dict(promptID=999, genre="fiction", label=0))
        a, stats = exporter.select_groups(rows, "multi_nli", "train", 15, 7)
        b, _ = exporter.select_groups(list(reversed(rows)), "multi_nli", "train", 15, 7)
        self.assertEqual(a, b)
        self.assertEqual(stats["excluded_fiction_or_unlabelled"], 1)
        groups = {row["promptID"] for row in a}
        self.assertEqual(len(groups), 5)
        for group in groups:
            self.assertEqual(sum(row["promptID"] == group for row in a), 3)

    def test_duplicate_choices_are_recorded_not_silently_relabelled(self):
        rows = [dict(id="bad", choices=dict(text=["x", "x"], label=["A", "B"]), answerKey="A"),
                dict(id="good", choices=dict(text=["x", "y"], label=["A", "B"]), answerKey="B")]
        selected, stats = exporter.select_groups(rows, "commonsense_qa", "train", 2, 7)
        self.assertEqual([r["id"] for r in selected], ["good"])
        self.assertEqual(stats["excluded_invalid_choices"], [dict(source_id="bad", reason="ambiguous_or_invalid_single_choice")])


if __name__ == "__main__":
    unittest.main()
