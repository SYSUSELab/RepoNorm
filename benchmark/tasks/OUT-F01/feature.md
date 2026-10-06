**Title**: Replace the XGrammar wrapper with a stateful low-level logits processor

**Description**:
Replace the wrapper around `xgr.contrib.hf.LogitsProcessor` with a custom
`XGrammarLogitsProcessor` built on XGrammar's matcher and token-bitmask APIs.
The scored runtime target is Torch/Transformers. The processor must support a
matcher per batch row, update each matcher after the first generated token,
apply the grammar bitmask to logits, and reset all per-generation state.

**API requirements**:

```python
class XGrammarLogitsProcessor(OutlinesLogitsProcessor):
    def __init__(self, compiled_grammar: str, tensor_library_name: str): ...
    def reset(self): ...
    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType: ...
```

The read-only attributes `compiled_grammar`, `tensor_library_name`, and
`is_first_token` must remain available. `XGrammarBackend` must select the
Transformers model's tensor library, construct tokenizer information, and pass
the selected library to processors created for JSON schema, regex, and CFG.
Existing Transformers behavior and public method signatures must remain
compatible. Unsupported tensor libraries raise `NotImplementedError` with
`"Library {library_name} is not available"`.

MLX execution is outside this corrected task's scored contract because the
frozen Linux image cannot execute the MLX runtime. No skipped MLX test counts as
evidence that MLX behavior works.
