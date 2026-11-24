**Title**: Add Opt-In StreamListener Trace Logging

`StreamListener` accepts `debug: bool = False` and
`debug_logger: logging.Logger | None = None`. The default logger is
`logging.getLogger("dspy.streaming.listener")`. The feature does not configure
global logging and does not change streaming output or boundary detection.

When debug is enabled and the logger accepts `DEBUG`, emit these public trace
events with exactly the shown punctuation and spacing:

- `Start detection: adapter=ChatAdapter, field='answer', stream_start=True, buffer_preview='...'`
- `Rolling end check: adapter=ChatAdapter, field='answer', reason='regex_match', buffered_size=25`
- `Emit chunk: len(token)=12, queue_size=3, is_last_chunk=False`
- `Flush: adapter=ChatAdapter, field='answer', truncated_buffer_length=15`
- `State reset for field 'answer' (allow_reuse=True)`

Use the actual adapter, field, sizes, and state in each event. Start, rolling
end, every emitted response, flush, and reusable reset remain observable.
A preview of at most 80 characters is included: content of length 80 or less is
unchanged; longer content uses its first 77 characters plus `...`.

Message interpolation and preview construction are lazy. If `debug=False` or
the logger rejects `DEBUG`, do not construct trace messages/previews and do not
call `logger.debug`.

