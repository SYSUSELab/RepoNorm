# Flexible ToolCalls input containers

Allow `dspy.ToolCalls` to validate the following equivalent inputs:

- The existing `{"tool_calls": [{"name": "foo", "args": {...}}, ...]}` container.
- A single `{"name": "foo", "args": {...}}` object.
- A bare list of these objects, including an empty list.
- Plain JSON strings representing any of those inputs.

Normalize them into the existing ToolCalls model while preserving call order,
argument values and nested structures. Do not mutate caller-owned input objects.
Keep the existing `from_dict_list`, already constructed ToolCalls inputs and
canonical model serialization behavior. In particular, `format()` retains its
existing list envelope containing `type: "tool_calls"`; this feature does not
change the output schema.

Expose `ToolCalls.validate_input(data)` as a normalization helper. It returns
ordinary normalized structures, not a new ToolCalls instance. JSON text is parsed
and normalized; non-JSON strings are returned unchanged by the helper and rejected
by model validation. Existing model instances may be passed through unchanged.
Do not execute input text. Malformed containers, missing fields and non-dictionary
arguments remain validation errors through the model API.

The supported shapes must work through `model_validate`, `model_validate_json`
and ordinary ChatAdapter parsing of a ToolCalls output field, without requiring
callers to normalize the input first.
