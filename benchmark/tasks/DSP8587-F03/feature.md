**Title**: Report Metrics for the Active Field Stream

Add `StreamListener.stats()`. It returns a serialization-friendly dictionary
with exactly these keys:

- `predict_name` and `field`
- `chunk_count` and `char_count`
- `first_ts` and `last_ts`
- `duration_s` and `avg_chunk_chars`

The counts describe responses emitted by `receive()` for the currently active
field stream. `char_count` is the sum of emitted `chunk` lengths. The first and
last timestamps use seconds; duration is their difference. Before an emission,
all counters are zero, timestamps and duration are `None`, and the average is
zero.

When the listener enters its ended state, all active-stream counters and
timestamps reset immediately, independently of `allow_reuse`. A final response
may still be returned, but `stats()` already reports the zeroed state after
that transition. Starting a reused stream also begins from zero. Metrics do
not alter buffering, boundary detection, response content, or LM usage
accounting.

