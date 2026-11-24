# Configurable instruction history limit

Add max_instruct_history=5 to GroundedProposer. Accept integers, clamp them to
0 through 20, and reject booleans and non-integers with ValueError. Zero includes
no prior instructions.

Apply the configured limit to the actual history text supplied for every predictor
and proposal. Preserve the existing aggregation and ordering: group identical
instructions, average their scores, select the highest-scoring N distinct
instructions and render that selected set in the existing ascending-score order.
Do not replace this with recency order or change unrelated proposal behavior.
Existing callers keep the default limit of five.
