import unittest

import numpy as np

from adacgr import AdaptiveStageController, MeanCUSUMSignal, PairedUtilitySignal


class MeanCUSUMTests(unittest.TestCase):
    def make_signal(self):
        signal = MeanCUSUMSignal(threshold=1.0, tolerance=0.1)
        signal.calibrate([1.0, 1.1, 0.9, 1.0])
        return signal

    def test_decision_uses_fixed_prior_reference_within_stage(self):
        signal = self.make_signal()
        prior = signal.reference()
        fired, standardized = signal.observe([1.5, 1.6])

        self.assertGreater(standardized, 0.0)
        self.assertTrue(fired)
        self.assertEqual(signal.reference(), prior)
        self.assertEqual(signal.last_observation["reference_before"], prior)
        self.assertEqual(signal.last_observation["reference_after"], prior)

    def test_post_fit_calibration_renews_reference_and_resets_evidence(self):
        signal = self.make_signal()
        signal.observe([1.5, 1.6])
        signal.calibrate([0.8, 0.9, 1.0])

        self.assertEqual(signal.evidence, 0.0)
        self.assertAlmostEqual(signal.reference()["mu"], 0.9)
        self.assertEqual(signal.reference()["count"], 3)

    def test_empty_batch_does_not_change_reference_or_evidence(self):
        signal = self.make_signal()
        reference = signal.reference()
        fired, standardized = signal.observe([])

        self.assertFalse(fired)
        self.assertEqual(standardized, 0.0)
        self.assertEqual(signal.reference(), reference)
        self.assertEqual(signal.evidence, 0.0)

    def test_nonfinite_observations_are_rejected_before_state_change(self):
        signal = self.make_signal()
        reference = signal.reference()

        with self.assertRaises(FloatingPointError):
            signal.observe([np.nan])

        self.assertEqual(signal.reference(), reference)
        self.assertEqual(signal.evidence, 0.0)


class StageControllerTests(unittest.TestCase):
    def test_feedback_and_budget_guards_gate_a_fired_clock(self):
        signal = MeanCUSUMSignal(threshold=0.1, tolerance=0.0)
        signal.calibrate([1.0, 1.0])
        controller = AdaptiveStageController(signal, min_feedback=4, min_budget=2)

        decision = controller.observe_batch(
            [2.0, 2.0],
            feedback_count=2,
            eligible_count=2,
            available_steps=1,
        )
        self.assertTrue(decision.fired)
        self.assertFalse(decision.accepted)

        decision = controller.observe_batch(
            [2.0, 2.0],
            feedback_count=2,
            eligible_count=2,
            available_steps=2,
        )
        self.assertTrue(decision.accepted)
        controller.complete_fit([1.0, 1.0])
        self.assertEqual(controller.feedback_count, 0)
        self.assertEqual(signal.evidence, 0.0)
        self.assertEqual(signal.reference()["mu"], 1.0)


class UtilitySignalTests(unittest.TestCase):
    def test_paired_gain_accumulates_and_reset_is_explicit(self):
        signal = PairedUtilitySignal(threshold=1.0, tolerance=0.0)
        replace, gain = signal.observe(
            [1.0, 1.0, 1.0],
            [0.8, 0.8, 0.8],
            [10, 11, 12],
        )

        self.assertTrue(replace)
        self.assertGreater(gain, 0.0)
        self.assertGreater(signal.evidence, 0.0)
        signal.reset_after_replacement()
        self.assertEqual(signal.evidence, 0.0)

    def test_paired_inputs_must_align(self):
        signal = PairedUtilitySignal()

        with self.assertRaises(ValueError):
            signal.observe([1.0], [0.9, 0.8], [1])

    def test_nonfinite_paired_losses_are_rejected(self):
        signal = PairedUtilitySignal()

        with self.assertRaises(FloatingPointError):
            signal.observe([1.0], [np.inf], [1])


if __name__ == "__main__":
    unittest.main()
