**Title**: Add an `IsEmail` validator

Add a public `IsEmail` validator that follows dirty-equals' existing equality,
assertion, and top-level import behavior. It accepts an optional `domain`
filter.

**Public contract**:

- Without a domain filter, accept conventional string email addresses with a
  non-empty local part and a dotted domain; reject malformed and non-string
  values.
- The usual local-part characters used by addresses such as
  `test.email+tag@domain.co.uk` and domains containing hyphens are supported.
- A supplied domain must match the complete domain exactly and
  case-sensitively; a subdomain is not an implicit match.
- Its representations are `IsEmail(*)` without a filter and, for example,
  `IsEmail('example.com')` with one.

For this validator, the supported grammar is deliberately finite, not full
RFC email syntax. The local part uses ASCII letters/digits and `_ % + -`,
with dots only between nonempty runs (no leading, trailing or consecutive
dots). Domain labels use ASCII letters/digits and internal hyphens; labels
are nonempty and cannot start or end with a hyphen. There are at least two
labels and the final label consists of at least two ASCII letters. Match the
entire string without trimming whitespace. Quoted addresses, Unicode labels
and delivery checks are outside scope. Fresh repr examples above do not
override normal successful-comparison value/repr behavior.
