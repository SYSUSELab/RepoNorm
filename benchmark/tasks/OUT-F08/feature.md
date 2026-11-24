**Title**: Bound XGrammar memory history without losing lifetime statistics

Add opt-in RSS-delta tracking to `XGrammarLogitsProcessor` and its JSON-schema, regex, and CFG factories, through `track_memory_usage: bool = False` and `memory_history_limit: int | None = None`.

For each successful processing call while tracking, sample process RSS immediately before and after processing. Record `max(after - before, 0)` in MB (1024 squared bytes). If either sampling attempt fails, record a zero delta for that call; generation must continue. Underlying processing exceptions propagate and do not count as successful calls. When disabled, tracking makes no RSS sampling calls.

`get_memory_stats()` returns an independent dictionary with an independently copied `memory_history` list and the following fields:

- `total_calls`: all successful tracked calls since reset;
- `peak_memory_mb`: largest delta across those calls;
- `avg_memory_per_call_mb`: arithmetic mean across all those calls, or zero initially;
- `memory_history`: retained deltas in call-completion order.

With `memory_history_limit=None`, retain all deltas. A non-negative integer N retains only the most recent N; zero retains no detail. Booleans and other invalid limits raise `ValueError`. Peak, total, and average still describe every successful call, not merely the retained window. `reset()` clears generation state and all accumulated statistics while preserving configuration.

Statistics updates and returned snapshots must be thread-safe. This does not require XGrammar's stateful grammar matching itself to support concurrent generation, or reset to race safely with an ongoing call. Factories must forward both options and retain default behavior when tracking is disabled.
