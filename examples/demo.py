"""Small synthetic demonstration of the two public policy components."""

import numpy as np

from adacgr import AdaptiveStageController, MeanCUSUMSignal, PairedUtilitySignal


# A prefix initializes the causal reference. No real dataset or model is needed
# to exercise the method-level policy.
rng = np.random.default_rng(7)
clock = MeanCUSUMSignal(threshold=2.0, tolerance=0.1)
clock.calibrate(rng.normal(1.0, 0.08, size=32))
controller = AdaptiveStageController(clock, min_feedback=8, min_budget=4)

for batch in range(1, 9):
    losses = rng.normal(1.0 if batch < 4 else 1.25, 0.08, size=8)
    decision = controller.observe_batch(
        losses,
        feedback_count=len(losses),
        eligible_count=len(losses),
        available_steps=8,
    )
    print("stage", decision)
    if decision.accepted:
        print("accepted boundary; caller would fit here")
        post_fit_tail = rng.normal(1.0, 0.08, size=8)
        controller.complete_fit(post_fit_tail)

utility = PairedUtilitySignal(threshold=0.5, tolerance=0.05)
incumbent = np.array([1.0, 1.1, 0.9, 1.2])
candidate = incumbent - 0.08
replace, gain = utility.observe(incumbent, candidate, users=[1, 2, 3, 4])
print("replacement", replace, "gain", round(gain, 4))
if replace:
    utility.reset_after_replacement()
