**Title**: Add custom TensorAdapter injection to XGrammar processors

**Description**:
Allow callers to inject a `TensorAdapter` at the existing Outlines processor
boundary by selecting `tensor_library_name="custom"`. This task covers adapter
selection, use by the shared `OutlinesLogitsProcessor` shape-normalization path,
and parameter forwarding through all XGrammar backend factory methods. It does
not claim that XGrammar's Torch-specific underlying processor can execute an
arbitrary proprietary tensor runtime.

**API requirements**:

```python
XGrammarLogitsProcessor(
    compiled_grammar: str,
    tensor_library_name: str = "torch",
    custom_adapter: Optional[TensorAdapter] = None,
)
```

The JSON schema, regex, and CFG factory methods accept and forward the same
arguments. Selecting `"custom"` requires an adapter. Supplying an adapter with
any other library is invalid. Both invalid combinations raise `ValueError`
during construction. The selected custom adapter must become the processor's
actual `tensor_adapter` and be used through its public call interface.

Existing built-in tensor adapters and default Torch behavior remain compatible.
