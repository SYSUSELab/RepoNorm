**Title:** Proactively invalidate FileSystemLoader template caches

`FileSystemLoader(..., auto_reload=True)` enables a polling monitor with a 0.5 second interval. It starts when an Environment first loads through the loader, tracks bound environments weakly, and proactively removes cache entries for modified, deleted, and added paths. This complements Environment's existing request-time `uptodate` check. Loader and Environment `auto_reload` are independent: monitoring can invalidate an Environment cache even when that Environment disables its lazy check.

`start_monitoring()` and `close()` are public and idempotent. A closed loader may be explicitly restarted; the loader is also a context manager. `auto_reload=False` never tracks environments, scans, or starts a thread, including when `start_monitoring()` is called. Background filesystem errors do not terminate the monitor.

`scan_once()` performs a synchronous comparison, invalidates changed entries, and returns canonical `/`-separated template names. `invalidate_changed(paths)` synchronously invalidates the supplied absolute paths and returns their canonical names. Paths outside search roots are ignored. Nested paths and leading-dot names are preserved. Closing terminates the thread, and weak tracking does not keep environments alive.
