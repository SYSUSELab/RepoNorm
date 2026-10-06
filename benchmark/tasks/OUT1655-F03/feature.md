**Title**: Add an Octal String Regex Type

Define and export `octal_str` from `outlines.types` as a `Regex` DSL term.
It matches the exact lowercase prefix `0o` followed by one or more ASCII digits
from `0` through `7`. Leading zeroes in the digit portion are allowed. Uppercase
`0O`, signs, underscores, whitespace, digits `8` or `9`, and a prefix without
digits are invalid. This is a deliberately restricted textual format rather than
every spelling accepted by Python source syntax.
