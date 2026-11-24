# Template path validation

Add and export `jinja2.loaders.validate_template_path(template: str) -> None`.
It returns `None` for a valid name and raises `TemplateNotFound(template)` when
the value is `None` or the empty string, contains a NUL byte, begins with `/` or
`\`, begins with an ASCII drive prefix such as `C:`, or contains an ASCII control
character below code point 32 other than tab, newline, or carriage return.
Those three named control characters remain valid.

Every built-in loader validates names before loading, including direct public
`get_source` and overridden `load` entry points. This covers FileSystemLoader,
PackageLoader, DictLoader, FunctionLoader, PrefixLoader, ChoiceLoader, and
ModuleLoader. An invalid DictLoader key or a FunctionLoader result must not bypass
validation merely because a source exists. Composite loaders preserve the
original invalid request in `TemplateNotFound` and do not swallow the validation
failure as a successful fallback.

The public `split_template_path` function follows the same validation contract.
Existing additional filesystem traversal checks and valid-name behavior remain
in place.
