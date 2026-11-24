**Title**: Add temperature scaling to XGrammar processors

Allow `XGrammarLogitsProcessor` and the JSON-schema, regex, and CFG factory
methods on `XGrammarBackend` to accept a `temperature` value.

**Public contract**:

- The default is `1.0`, preserving existing behavior. The configured value is
  exposed as `processor.temperature` and remains unchanged by `reset()`.
- Temperature must be a finite numeric value greater than zero; invalid values
  raise `ValueError`.
- Before grammar constraints are applied, logits are divided by the
  temperature. Grammar validity and the existing output type and shape are
  preserved.
- All three backend factory methods forward the option consistently, and calls
  that omit it remain backward compatible.
