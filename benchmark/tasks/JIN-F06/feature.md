**Title**: Transform displayed group keys with sync or async callbacks

Add key_transform=None to groupby. A supplied callback receives each completed
group's original key once and returns its displayed grouper. Original keys still
determine grouping and sorting; display-key collisions must not merge groups.
None leaves keys unchanged. Preserve existing attribute/default behavior.

In both execution modes, ordinary callbacks and @pass_environment callbacks are
supported; decorated callbacks receive the active Environment before the key.
Async environments additionally await callback results when necessary and accept
asynchronous input iterators. Synchronous mode need not execute async callbacks.
Callback exceptions propagate unchanged. Returned plain strings and Markup retain
their normal template escaping semantics. Group members and their order are not
changed by the display transformation; empty input calls no callback.
