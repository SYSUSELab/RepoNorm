**Title**: Add a fallback processor to XGrammar

Allow `XGrammarLogitsProcessor` and the XGrammar backend's JSON-schema, regex,
and CFG factory methods to accept an optional compatible `fallback_processor`.

**Public contract**:

- Without an error, the primary grammar processor is used and the fallback is
  untouched.
- If primary logits processing raises an exception, the same call is completed
  through the fallback and a warning records the switch.
- Once activated, the fallback handles subsequent calls without repeatedly
  retrying the failed primary processor.
- `reset()` resets both processors as applicable and makes the primary eligible
  for the next call again.
- If no fallback was supplied, the original primary exception propagates.
- Existing construction and backend calls that omit the parameter retain their
  behavior.
