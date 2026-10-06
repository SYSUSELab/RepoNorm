**Title**: Add an `IsHash` validator

Add a public `IsHash(algorithm)` validator that follows dirty-equals' existing
equality, assertion, and top-level import behavior.

**Public contract**:

- Support MD5, SHA1, SHA224, SHA256, SHA384, and SHA512 names
  case-insensitively.
- A matching value is a string with exactly the hexadecimal length defined by
  the selected algorithm. Upper-, lower-, and mixed-case hexadecimal digits are
  valid; non-string, non-hexadecimal, shorter, and longer values compare
  unequal.
- An unsupported algorithm raises `ValueError` at construction time.
- The representation preserves the spelling supplied by the caller, for
  example `IsHash('md5')` and `IsHash('SHA256')`.

Validation concerns only the complete hexadecimal representation, not whether
it is the digest of a real file. Whitespace (including a final newline) is
not hexadecimal. Repr examples refer to fresh validators; normal successful
comparison value/repr behavior remains unchanged.
