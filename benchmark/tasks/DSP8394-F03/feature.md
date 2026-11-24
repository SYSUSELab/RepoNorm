# Optional cache entry lifetimes

Add `ttl=None` to `Cache` and `dspy.clients.configure_cache`, and expose
`dspy.clients.set_ttl(seconds)` for the active global DSPy cache.

A TTL is a positive finite number of seconds; `None` means no expiration.
Reject zero, negative, nonfinite, boolean and nonnumeric TTLs with `ValueError`.
TTL is optional and existing callers retain their cache behavior.

Each successful write starts that entry's lifetime. Reads, including promotion
from disk into memory, must not restart it. Once the deadline passes, reads
must miss in every enabled layer. Overwriting an entry starts a new lifetime.
Support memory-only, disk-only, combined and disabled cache configurations.
Disk entries retain their expiry when another Cache instance opens the same store.

`set_ttl` changes the lifetime used by subsequent writes without clearing entries,
extending their existing deadlines, or changing the active cache's enabled layers,
location or capacity. Setting it back to `None` makes new writes nonexpiring;
already expiring entries still expire. Invalid updates leave the cache unchanged.

Keep existing cache value copying, usage clearing, key handling and request-cache
behavior. Saving and loading memory entries must not renew their deadlines.
No particular in-memory container or expiration algorithm is required.
