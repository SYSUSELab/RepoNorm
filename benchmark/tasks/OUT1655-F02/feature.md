**Title**: Add a Binary String Regex Type

Define and export `binary_str` from `outlines.types` as a `Regex` DSL term.
It matches the exact lowercase prefix `0b` followed by one or more ASCII `0`
or `1` digits. Uppercase `0B`, signs, underscores, whitespace, other digits,
and a prefix without digits are invalid. The term must retain normal Pydantic
schema and regex DSL behavior.
