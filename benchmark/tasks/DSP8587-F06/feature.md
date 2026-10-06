**Title**: Invoke a Callback for Each Emitted Streaming Response

Add
`on_chunk: Callable[[StreamResponse], None] | None = None`
to `StreamListener`. The default preserves existing behavior.

Whenever `receive()` creates a `StreamResponse`, invoke `on_chunk(response)`
immediately and exactly once before returning that same response object.
This includes ordinary and final responses. If a separately combined timeout
feature emits a response through an analogous emission point, the same rule
applies there.

The callback return value is ignored. Do not catch callback exceptions; the
original exception propagates to the caller. This feature does not change
`StreamResponse` fields, buffering, adapter boundaries, or which model chunks
produce responses.

