**Title:** Add observable priority fallback to `trans`

`trans` accepts one `priority=<expression>` parameter in any normal header position. Its value is converted with `int`; invalid, `None`, and negative values become 0. At most ten levels are attempted: level 0 through `min(converted_priority, 9)`. A duplicate priority is a `TemplateAssertionError`.

`Environment.install_gettext_callables` accepts optional `priority_gettext(message, level)` and `priority_ngettext(singular, plural, n, level)` callables. When the matching resolver is installed, levels are tried in increasing order. A result equal to the applicable source message is a miss; the first different result stops the search. If no priority resolver is installed, the ordinary gettext or ngettext callable is invoked exactly once.

Priority works for singular and plural messages, variables, trimmed/notrimmed, old/new-style formatting, and autoescaping. Blocks without priority keep existing behavior. Reinstalling or uninstalling callables replaces or removes the priority resolvers.
