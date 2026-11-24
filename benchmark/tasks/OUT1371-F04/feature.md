**Title**: Add Selectable Built-In Prompt Filter Categories

Add an optional `builtin_filters` list to `@outlines.prompt`, `Prompt.from_str`,
and `Prompt.from_file`. It selects any of three finite categories. The existing
bare decorator and `None` or an empty list retain default Jinja and Outlines prompt
filters. Repeating a category is harmless. An unknown category, including the empty
string, raises `ValueError("Unknown builtin filter category: <name>")` while the
prompt is constructed.

`string_utils` provides:

- `upper`, `lower`, and `title`, converting the input to text;
- `reverse`, reversing the text by code point;
- `truncate(value, length, end="...")`, returning unchanged text when it fits and
  otherwise producing at most `length` characters by replacing the tail with as
  much of `end` as fits; a negative length raises `ValueError`;
- `pad_left(value, width, fill=" ")` and `pad_right(...)`, using Python string
  left/right padding semantics, including the one-character fill requirement.

`math_utils` provides `add`, `subtract`, `multiply`, and `divide` as Python binary
operations; `round(value, ndigits=None)` as Python rounding; `abs`; and
`min(value, *others)` / `max(value, *others)`.

`format_utils` provides:

- `json`, using `json.dumps` defaults;
- `csv`, formatting an iterable as one CSV row with standard quoting and no line
  terminator;
- `indent(value, width=4)`, adding `width` spaces to each non-empty line;
- `wrap(value, width=70)`, wrapping text with Python's normal word wrapping;
- `escape` and `unescape`, applying HTML escaping (including quotes) and unescaping.

Selected filters work in decorator, string, and file prompts, including file
includes and inheritance. Existing Outlines-specific filters remain available.
The implementation must provide every filter listed above; YAML is not part of
this feature.
