**Title**: Bound and safely share compiled grammars in `XGrammarBackend`

Add `cache_compiled_grammars: bool = False` and `max_cached_grammars: int | None = None` to `XGrammarBackend`.

When caching is disabled, every factory request compiles normally. When enabled, requests with the same grammar kind and identical source string reuse a successful compilation on that backend instance; kinds and instances remain isolated. Each request still returns a fresh, independent stateful `XGrammarLogitsProcessor`.

`None` permits an unbounded cache; otherwise the limit must be a positive integer (not a boolean), with invalid limits raising `ValueError`. Keep at most that many completed grammars, evicting the least recently used successful entry. A hit refreshes recency. No canonicalization of different source strings is required.

Concurrent cold requests for the same key must share an in-progress compilation. If it fails, waiting requests receive that compilation exception, no failed entry is cached, and a later request may retry. Different keys may be compiled serially or concurrently. Preserve JSON-schema, regex, and CFG factory behavior and independent generation/reset state.
