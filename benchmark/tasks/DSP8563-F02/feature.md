# Parse a safe subset of Python-style tool calls

Add public `parse_python_calls(text: str) -> dict` and support the same string
input through `ToolCalls.validate_input` and `ToolCalls.model_validate`.
The parser returns `{"tool_calls": [{"name": ..., "args": {...}}, ...]}`.

Accept one or more top-level calls separated by newlines or semicolons, preserving
source order. A callee is an identifier or a dot-separated sequence of identifiers;
use its final identifier as the tool name. A dotted name is only a name, never an
attribute lookup or an evaluated expression. Calls may have no arguments or only
explicit keyword arguments.

Argument values may contain strings (including Unicode), finite integers/floats,
booleans, None, and nested lists, tuples and dictionaries. Dictionary keys must
be strings. Convert tuples recursively into lists. Unary + and - are permitted
only on numbers, not booleans. Preserve all other literal values.

Reject positional arguments, argument unpacking, duplicate keyword arguments or
dictionary keys, variables, computed/attribute values, nested calls, f-strings,
comprehensions, lambdas, bytes, complex numbers, Ellipsis and non-finite numbers.
Reject statements other than calls and reject Python comments; a # inside a
string literal is ordinary string content. Empty input and malformed syntax are
errors. Raise ValueError with a useful reason; exact error wording is unrestricted.

Never execute the supplied text or any expression within it. Preserve existing
non-string ToolCalls model input and output formatting. The choice of parser,
private helpers and implementation files is unrestricted.
