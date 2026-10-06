# Batch multiple option values

Add `batch_size: int | None = None` to `click.Option` and `@click.option`. It is valid only with `multiple=True`. After normal type conversion and callback processing, group the final ordered tuple into an outer tuple of non-empty tuple batches of at most `batch_size` items. Preserve every value, including `0`, `False`, the empty string, and `None`; preserve input order and include a shorter final batch. Empty input produces `()`.

`batch_size` must be a positive integer and must reject booleans. A non-integer raises `TypeError`, a non-positive integer raises `ValueError`, and using it without `multiple=True` raises `TypeError` at option construction. `None` preserves existing Click behavior. This groups already-parsed values; it does not claim streaming input or lower parser memory use, and it does not require changing Click's shared `batch` helper.
