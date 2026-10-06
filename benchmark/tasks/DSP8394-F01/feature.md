# Temporarily bypass DSPy cache reads while retaining writes

Add `Cache.bypass()` as a synchronous context manager, available on the global
cache as `with dspy.cache.bypass():`. Within that context, reads behave as cache
misses even when a value is stored. Writes remain enabled according to the
existing memory/disk configuration. After the context ends, newly written values
can be read normally.

The context must work with the existing `request_cache` decorator for both
synchronous and asynchronous functions: a bypassed call recomputes its result,
stores it, and later ordinary calls reuse the fresh result. Entering bypass must
not clear existing entries or change cache keys, capacity, or enabled layers.

Nested contexts retain bypass until their outer context exits. Normal and
exceptional exits restore the previous state. Bypass is local to the current
thread/asynchronous execution context; an independently started thread or sibling
async task must retain its own state. The synchronous context manager can be used
inside an async function and must remain effective across awaits.

Keep existing cache operations and ordinary decorated calls compatible. The
choice of context-state representation, private helpers and file layout is yours.

Bypass is local to the Cache instance on which it is entered. Other Cache
instances in the same execution context continue to serve hits unless their own
bypass context is active. Independently nested bypass contexts restore each
instance's prior state when they exit.
