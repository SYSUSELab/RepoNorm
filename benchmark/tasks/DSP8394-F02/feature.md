# Experiment-scoped cache namespaces

Extend `dspy.configure_cache` (also available from `dspy.clients`) and
`request_cache` with an optional `namespace=None` argument, preserving their
existing parameters. Support `dspy.cache.set_namespace(value)`, assignment to
`cache.namespace`, and `with cache.namespace(value):` for temporary changes.
The namespace accessor compares equal to the current string or None, and exposes
that current value through `.value`.

Namespaces are strings (including the empty string) or None. Invalid values raise
ValueError. None uses the original cache keys; a string prefixes a key with the
first eight hexadecimal SHA-256 characters of the UTF-8 namespace and a colon.
This public key format is retained for compatibility. Both cache layers share the
same namespace and identical requests can be reused within the same namespace.

Namespace state belongs to the current execution context, with shared cache data.
Temporary contexts nest and restore the prior state even on exceptions. Bare
threads and independent async tasks isolate their changes. DSPy parallel
primitives inherit the caller's namespace and child changes must not leak into
siblings or back to the parent. This must work with ParallelExecutor at one or
multiple worker threads.

A non-None decorator namespace temporarily overrides the current one for the
whole synchronous or asynchronous call, including cache lookup and writeback;
it restores the caller's namespace on return or exception. With no decorator
namespace, use the caller's active namespace. configure_cache sets the initial
namespace for the calling context, including clearing it when None is supplied.
Existing non-namespaced cache behavior remains available.
