**Title**: Add a Social Security Number Format Type

Define and export `ssn` from `outlines.types` as a `Regex` DSL term. It matches
exactly three ASCII digits, a hyphen, two ASCII digits, a hyphen, and four ASCII
digits (`XXX-XX-XXXX`). This is format-only validation: values such as
`000-00-0000` and `999-99-9999` are accepted, and no issued-number or prohibited-
range rules are applied.

```python
from outlines import types

SsnModel = types.ssn
```
