# Instruction length limits and optional rephrasing

Add min_instr_chars=30, max_instr_chars=600 and rephrase_when_too_long=False to
GroundedProposer. Bounds must be non-boolean nonnegative integers with min <= max;
reject invalid bounds with ValueError.

Measure the generated instruction after its existing prefix stripping. If shorter
than min, use the predictor's canonical signature instructions. If longer than max
and rephrasing is disabled, retain the last sentence boundary (. ! or ? followed
by whitespace/end) that fits with one character reserved for …; otherwise retain
a hard prefix of max-1 characters and append …. Final length cannot exceed max;
a zero cap returns the empty string.

When rephrasing is enabled for an overlong generated instruction, make exactly one
ShortenInstruction call with long_instruction and max_chars inputs and a
proposed_instruction output using prefix "PROPOSED INSTRUCTION:". Strip its output
prefix; if it is empty or still shorter than min, fall back to the canonical basic
instruction. Hard-truncate a still-overlong rephrase or its fallback to max.
Generation errors propagate. Rephrasing must not mutate LM kwargs.

The basic fallback may itself be shorter than min; retain that explicit fallback
exception instead of fabricating text. It must still satisfy max. Do not retry
rephrasing. With verbose=True, report the fallback/trim/rephrase action with the
measured length and applicable threshold. Preserve the existing proposal flow.
