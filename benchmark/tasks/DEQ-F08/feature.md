**Title**: Add an `IsRegex` matcher

Add a public `IsRegex` matcher that can be imported from `dirty_equals` and
participates in the library's normal equality and assertion behavior.

**Public contract**:

- `IsRegex(pattern, flags=0, *, mode='match')` accepts a string pattern or a
  compiled string regular expression and optional standard `re` flags.
  Modes `match`, `search`, `fullmatch` have Python's corresponding semantics;
  the default matches at the beginning, not necessarily the entire string.
  Compiled patterns retain their flags. Passing one with nonzero external
  flags raises ValueError; invalid modes raise ValueError at construction.
  Bytes patterns raise TypeError and are outside scope.
- A string compares equal when it matches the supplied pattern; non-string
  values compare unequal.
- Empty strings, Unicode text, escaped metacharacters, multiline patterns, and
  combined flags must behave as Python regular expressions normally do.
- An invalid pattern raises `re.error` when the matcher is constructed.
- The representation mirrors a constructor call: without flags it is
  `IsRegex('pattern')`; with flags it includes the flags argument. Pattern
  escaping in the representation must be unambiguous.
- Non-default modes are distinguishable in fresh repr as `mode='search'`
  or `mode='fullmatch'`. Successful comparisons retain normal library repr
  and value behavior.
