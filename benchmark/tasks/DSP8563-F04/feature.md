# Stable ToolCall fingerprints

Give each `ToolCalls.ToolCall` read-only `.fingerprint` and `.id` metadata, available
when directly constructed or created through `ToolCalls.from_dict_list` and normal
model validation. Compute them once at construction; later name/args mutations do
not recalculate them. Callers cannot assign or override these computed attributes.
The eight-character id is a display abbreviation, not a uniqueness guarantee.

The fingerprint is the 64-character hexadecimal SHA-256 of the UTF-8 canonical JSON
object `{"name": name, "args": canonical_args}`, using sorted keys and compact
comma/colon separators with ordinary JSON ASCII escaping. In arguments, recursively
sort dictionary keys, treat tuples as lists and trim surrounding whitespace from
string values. Preserve numeric/boolean types and list order; do not trim dictionary
keys or the tool name. Do not modify the stored original argument values.

The supported argument domain is JSON values with string dictionary keys, plus
tuples treated as arrays. Reject nonfinite numbers and non-JSON values with
ValueError. Fingerprinting does not execute arguments.

Metadata is excluded from the existing model dump and execution format: keep the
current serialized name/args structures and ToolCalls.format list envelope.
No lookup, deduplication, collision handling or automatic metadata updates are needed.
