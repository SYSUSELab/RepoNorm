# Concurrent direct-file edit locks

Add `lock_files: bool = False` to `click.edit`. For a direct `filename` with locking enabled, atomically create `.{basename}.lock` in the same directory before launching the editor. The UTF-8 lock content has a `PID: <decimal>` line and a `Time: <ISO datetime>` line. An existing lock prevents editor launch and raises `ClickException` with the lock path and readable lock information when available. Same basenames in different directories have independent locks.

Remove the acquired lock after normal completion, nonzero editor exit, or process-launch failure. A creation or removal `OSError` raises `ClickException` and chains the original exception; do not silently claim successful cleanup. Locking does not apply to the temporary file used by the `text` form. `False` preserves existing behavior. Atomic exclusive creation is required, while the context manager, file-opening API, and private helpers are not prescribed.
