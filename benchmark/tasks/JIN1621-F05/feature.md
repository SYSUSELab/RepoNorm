**Title:** Add an optional bounded resolved-path cache to FileSystemLoader

`FileSystemLoader(..., path_cache_size=0)` may cache successful template-name to filename resolution. Zero disables caching and is the backward-compatible default. A positive integer enables least-recently-used eviction at that capacity; booleans and negative values are rejected.

Misses are never cached. A cached file that is removed is resolved again. If a template was cached from a lower-priority search path and the same name later appears in an earlier path, the earlier path takes precedence on the next load. These freshness rules apply independently of Environment's compiled-template cache and `auto_reload` option.

`clear_path_cache()` removes all entries. `resize_path_cache(capacity)` changes the bound immediately, evicts least-recently-used entries as needed, and disables and clears the cache at zero. Callers observe caching through resolution behavior only; no cache container is public.

`resolve_template_path(template)` is the public customization hook for an uncached priority-order lookup. It returns the resolved filename or `None`. Cache misses call this hook; valid cache hits do not, so subclasses can customize or observe resolution without depending on the cache's private storage.
