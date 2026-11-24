# Cache pure option type conversions

Add `cache: bool = False` to `click.Option` and `@click.option`. When enabled, the same Option instance may reuse a successful type-conversion result for the same hashable raw scalar value across invocations in the current process. Cache identity includes the Option instance, so equal names, values, and type names on different options never collide. Failed conversions are not stored. Callbacks and other post-conversion processing still run on every invocation.

Do not cache `File` or `Path` types, `multiple=True`, `nargs != 1`, `None`, or unhashable raw values. No filesystem freshness claim is made. Each Option exposes `clear_value_cache()` to discard its own entries; clearing one option does not affect another. `cache=False` preserves existing behavior. The task does not prescribe a global dictionary, digest algorithm, or key representation.
