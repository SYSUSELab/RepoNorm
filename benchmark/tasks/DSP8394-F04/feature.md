# Cache hit and miss statistics

Expose `cache.stats()` and `cache.reset_stats()` on Cache. stats returns a fresh
dictionary with integer keys `hits`, `mem_hits`, `disk_hits`, `misses`, initially
zero. Each get records exactly one outcome: memory hit, disk hit, or miss, including
failed key generation. hits equals mem_hits plus disk_hits. Writes do not count as
reads; disk-to-memory promotion does not count as an extra hit. Reset clears counters
without clearing entries. Counters remain accurate under concurrent access.

When `dspy.settings.context(track_usage=True)` is active, each synchronous Module
call or asynchronous acall that returns a Prediction attaches an independent cache
usage snapshot available through `Prediction.get_cache_usage()`. Nested calls each
capture their own events; the enclosing call includes its nested calls' events.
Tracking must work even inside an existing LM usage tracking context. Preserve the
existing LM usage behavior. Without tracking, new Predictions return None.

Per-call statistics are independent of global totals and later calls or resets.
Changing a returned statistics dictionary must not mutate stored counters or an
already captured Prediction. Concurrent async calls must not mix their individual
counts. Exceptions must restore the caller's tracking context. Keep existing cache
key, value, copy-on-read, usage-clearing and eviction behavior.
