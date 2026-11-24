**Title**: Mark the Final Emitted Streaming Chunk

Add `is_last_chunk: bool = False` to `dspy.streaming.StreamResponse`.
Every response emitted by `StreamListener.receive()` carries this flag.
Ordinary responses use `False`; the response containing the last content for
the configured field uses `True`.

The rule also applies when a Chat, JSON, or XML adapter returns the whole
response in one model chunk. If that response contains the configured field
and its normal end identifier, `receive()` emits the field's actual content
once with `is_last_chunk=True`. A full response for another field emits
nothing. Existing adapter boundary and whitespace behavior stays in effect.

Keep `predict_name`, `signature_field_name`, and `chunk` unchanged. The default
allows existing callers to construct a three-argument `StreamResponse`. With
`allow_reuse=True`, each completed input can emit its own final response after
the listener resets.

