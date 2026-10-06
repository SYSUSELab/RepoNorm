**Title**: Add an `IsMac` validator

Add a public `IsMac` validator that follows dirty-equals' existing equality,
assertion, and top-level import behavior.

**Public contract**:

- With no format argument, accept six-octet hexadecimal MAC addresses in
  colon-separated, hyphen-separated, three-group dotted, and compact forms.
- Matching hexadecimal digits is case-insensitive, but malformed lengths,
  non-hexadecimal characters, mixed separators, and non-string values compare
  unequal.
- The optional format is one of `colon`, `hyphen`, `dot`, or `compact`; when
  supplied, otherwise valid addresses in a different form compare unequal.
- Its representations are `IsMac(*)` without a format and, for example,
  `IsMac('colon')` with one.

The format must match the entire string: leading/trailing whitespace, including
a final newline, is not allowed. Representation examples describe a fresh
validator; after a successful comparison, retain the library's normal value
and representation behavior.
