# ToolCalls JSON inside Markdown fences

Make normal ToolCalls validation accept JSON wrapped in a Markdown code fence.
Support `ToolCalls.model_validate`, `model_validate_json` when its decoded value
is such text, and ChatAdapter parsing of a ToolCalls output field. Preserve the
existing output model and execution format.

First accept valid plain JSON without treating backticks inside JSON strings as
fences. Otherwise inspect the first triple-backtick block: its opening fence may
have any language label (including json, JSON, or no label), followed by a newline;
the next triple-backtick delimiter closes it. Parse the trimmed body as JSON,
ignoring surrounding prose. Only this first block is considered; do not skip an
invalid first block to accept a later one. Inline fences without that opening
newline, missing closures and malformed JSON are not supported.

Accept canonical tool_calls containers, a single name/args object and bare lists,
including empty lists, and apply the normal ToolCall field validation afterward.
Do not execute text or add YAML or Python-call parsing.

Keep `ToolCalls.validate_input(data)` as a public extraction helper: it returns
parsed Python JSON structures on success and the original input unchanged when
extraction fails. The normal model API must reject unsupported input with its
ordinary validation error. Calling the helper manually must not be necessary.
