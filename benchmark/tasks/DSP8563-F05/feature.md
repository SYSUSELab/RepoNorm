# Non-execution ToolCall metadata and JSON comments

Add optional ToolCalls.ToolCall attributes, defaulting to None: note, comment and
annotation (strings), tags (list of strings), priority (string or integer), and
meta (dictionary). Retain them through direct construction and from_dict_list.
Keep existing name/args validation and ignore unknown fields as before. Metadata
must not become Tool arguments or change execution payloads.

`format()` keeps the existing list envelope containing `type: "tool_calls"` and
only execution fields. Add `format_with_metadata()` with that same envelope and
non-None metadata beside each call's type/function fields, not inside arguments.
This applies whether or not metadata is present. Omit unset metadata; preserve
explicit empty strings/lists/dictionaries and zero priority.

Support `ToolCalls.from_string(text)` for JSON with Python-style # comments outside
quoted strings, on their own lines or after JSON content. Accept a single call,
a bare list or the canonical tool_calls container. Preserve # and escaped quotes
inside JSON strings. Combine nonblank comment bodies in source order, trim their
surrounding whitespace and join with newlines. As document-level annotations, the
combined comment applies to every call in the parsed input and overrides explicit
comment fields; note remains independent. Do not infer per-call association.

`from_dict_list(items, comments=None)` also accepts an optional list of comment
strings with the same combination rule. Empty/blank comments do not override an
explicit comment. Reject invalid JSON, invalid call/metadata fields and non-string
comment-list members with ValueError. Do not mutate input objects or execute text.
This task does not require fingerprint functionality from another feature.
