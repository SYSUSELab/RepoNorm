**Title**: Add Custom Jinja Filters to Prompts

Add an optional `filters` mapping to all three public prompt-construction APIs:

- `@outlines.prompt(filters={...})`
- `Prompt.from_str(content, filters={...})`
- `Prompt.from_file(path, filters={...})`

Each mapping key is the name available after `|` in a Jinja template, and its value
is the callable used for that filter. Filter callables may accept arguments supplied
by the template. The existing bare `@outlines.prompt` form must continue to work.

The supplied filters are local to the prompt being constructed. If a supplied name
matches a Jinja built-in filter or one of Outlines' predefined prompt filters (`name`,
`description`, `source`, `signature`, `schema`, or `args`), the supplied callable takes
precedence for that prompt. Passing `None`, an empty mapping, or omitting `filters`
keeps the existing default filters.

`Prompt.from_file` must retain its existing relative include and template-inheritance
behavior while making the custom filters available to every template loaded in that
environment.

Also add `Prompt.with_filters(filters)`, which returns a new callable Prompt derived
from an existing prompt. The supplied mapping extends its current filters and
overrides matching names, without changing the original prompt or other derived
prompts. Passing `None` or an empty mapping retains all current filters. Derivation
can be chained and must work for prompts created by any of the three APIs above,
preserving their call signatures, default arguments, and template source semantics
(including relative includes and inheritance for file templates).

Examples:

```python
@prompt
def my_prompt(text):
    '''{{ text }}'''

def surround(value, left, right):
    return f"{left}{value}{right}"

@prompt(filters={"surround": surround})
def tagged_prompt(text):
    '''{{ text | surround("<", ">") }}'''
```
