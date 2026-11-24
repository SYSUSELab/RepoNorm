# Weighted tip selection for instruction proposals

Add optional tip_weights=None to GroundedProposer. When random tip selection is
enabled, use these relative weights for the existing TIPS entries, and pass the
selected tip into the actual instruction generator. Missing keys have weight 1;
unknown keys are ignored. Positive finite integer/float weights are usable;
non-numeric, boolean, nonfinite and nonpositive values have weight zero.

Zero-weight entries must never be selected when any positive weight exists.
None and an all-zero effective mapping use uniform selection over existing tips.
Avoid overflow when summing large finite weights. Keep the existing RNG-based
reproducibility: the same seed and call sequence give the same selections, without
requiring a particular sampling method or private helper. Preserve the existing
use_tip behavior, including selection of the empty "none" tip, and leave random
selection disabled when set_tip_randomly=False.
