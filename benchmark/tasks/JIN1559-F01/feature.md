# Context on translation blocks

Allow a string literal immediately after `{% trans` to specify translation
context. A singular block with context uses the installed `pgettext(context,
message)` callable. A plural block uses `npgettext(context, singular, plural,
n)`. Context is a compile-time string literal; it is not a render expression.

Context blocks preserve the existing variable interpolation, plural selector,
`trimmed` and `notrimmed`, old-style and new-style gettext, and autoescaping
behavior. A block without the leading context literal continues to use `gettext`
or `ngettext` unchanged.

The public `Environment.extract_translations` interface reports contextual block
messages as `pgettext` or `npgettext` calls, including the context and both plural
forms where applicable. The feature does not prescribe generated AST node shapes
or parser helper names.
