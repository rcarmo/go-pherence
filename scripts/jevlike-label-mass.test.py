#!/usr/bin/env python3
"""Pure-Python math tests for the offline label-mass diagnostic."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("jevlike_label_mass", Path(__file__).with_name("jevlike-label-mass.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LabelMassMathTest(unittest.TestCase):
    def test_module_import_does_not_load_torch(self):
        self.assertNotIn("torch", MODULE.__dict__)

    def test_logsumexp64_shift_invariance(self):
        values = [-1000.0, -1001.0, -1002.0]
        shifted = [value + 1234.5 for value in values]
        self.assertAlmostEqual(MODULE.logsumexp64(shifted) - MODULE.logsumexp64(values), 1234.5, places=12)

    def test_underflow_keeps_log_allowed_mass_finite(self):
        report = MODULE.compute_allowed_label_mass([0.0, -1000.0, -1001.0], [1, 2])
        self.assertTrue(math.isfinite(report["log_allowed_mass"]))
        self.assertEqual(report["allowed_mass"], 0.0)
        self.assertAlmostEqual(sum(report["conditional_probabilities"]), 1.0, places=13)

    def test_shift_invariant_allowed_mass_and_conditionals(self):
        base = MODULE.compute_allowed_label_mass([2.0, 1.0, -3.0, -4.0], [0, 1])
        shifted = MODULE.compute_allowed_label_mass([202.0, 201.0, 197.0, 196.0], [0, 1])
        self.assertAlmostEqual(base["log_allowed_mass"], shifted["log_allowed_mass"], places=12)
        self.assertAlmostEqual(base["allowed_mass"], shifted["allowed_mass"], places=12)
        for left, right in zip(base["conditional_probabilities"], shifted["conditional_probabilities"]):
            self.assertAlmostEqual(left, right, places=12)

    def test_descriptive_flag_is_confidence_and_mass_only(self):
        report = MODULE.compute_allowed_label_mass([10.0, 0.0, 12.0, 11.0], [0, 1])
        self.assertTrue(report["confidence"] >= 0.9)
        self.assertTrue(report["allowed_mass"] < 0.1)
        self.assertTrue(report["diagnostic_high_conditional_low_mass"])

    def test_rejects_nonfinite_or_malformed_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.logsumexp64([0.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            MODULE.compute_allowed_label_mass([0.0], [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.compute_allowed_label_mass([0.0, 1.0], [1, 1])
        with self.assertRaisesRegex(ValueError, "out of range"):
            MODULE.compute_allowed_label_mass([0.0, 1.0], [2])


if __name__ == "__main__":
    unittest.main()
