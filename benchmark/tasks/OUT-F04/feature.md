**Title**: Configure XGrammar generation batch limits through every factory

Add `max_batch_size: int | None = None` to `XGrammarLogitsProcessor` and all three `XGrammarBackend` processor factories (JSON schema, regex, CFG). `None` means unlimited. A provided value must be a non-negative integer, excluding booleans; otherwise construction raises `ValueError`. Zero is valid and disallows any nonempty generation.

Keep the configured value available as `processor.max_batch_size`. Oversized-batch errors retain the message `Batch size {batch_size} exceeds maximum allowed batch size {max_batch_size}`.

Reject an initial generation batch larger than the limit with `ValueError` before initializing generation state. A rejected first call must leave the processor usable for a later valid call. Through the public call interface, an unbatched sequence counts as batch size one. Valid batched and unbatched calls preserve their input/output shape convention and grammar-constraining behavior.

`reset()` clears generation state, preserves the configured limit, and causes the next generation to be checked again. The limit must actually apply to processors returned by all three factories, not merely be accepted as an unused argument. Changing the batch size within one ongoing generation remains outside this task.
