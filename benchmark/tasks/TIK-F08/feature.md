**Title**: Add Token Pattern Tracking

Add `return_repeated_pattern=False` to `Encoding.encode`. When true, return
`(tokens, patterns)`, where tokens are exactly the ordinary encoded sequence
and patterns maps up to three distinct adjacent token-ID pairs to their
occurrence counts. Count overlapping pairs too: three consecutive copies
of token A contribute two occurrences of (A,A). Include pairs occurring only
once if they belong in the top three. Empty or single-token sequences have
no pairs. If fewer than three distinct pairs exist, return all of them.

At the top-three boundary any selection among equal-frequency pairs is
valid; dictionary insertion order is not scored. Existing special-token and
Unicode handling remains applicable. False or omitted retains the original
list return. This feature supports corpus inspection; no timing or memory
performance guarantee is required.
