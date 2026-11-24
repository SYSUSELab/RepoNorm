**Title:** Add an optional bounded cache for installed translation callables

`Environment.install_gettext_callables` accepts `cache_size=0` and `cache_partition=None`. Zero is the backward-compatible default and performs every provider call. A positive integer caches raw results of gettext, ngettext, pgettext, and npgettext independently using a bounded least-recently-used cache. Keys include the function kind, context where applicable, singular and plural messages, plural count, and the caller-supplied hashable partition.

Applications whose translation result depends on locale or request state must give each such state a distinct partition. `set_translation_cache_partition(value)` changes it and clears old entries. `clear_translation_cache()` clears entries; `resize_translation_cache(capacity)` applies a new non-negative bound immediately, with zero disabling and clearing the cache. Invalid capacities raise `ValueError`.

Installing or uninstalling translation callables clears prior results. Caching works before old/new-style formatting and autoescaping, so formatting variables are not part of raw provider keys. Each Environment and overlay has independent cache state.
