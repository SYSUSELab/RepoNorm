**Title:** Add conditional translation blocks with lazy fallback bodies

The `ConditionalI18nExtension` provides `{% ctrans condition, ... %}` followed by a required `{% fallback %}` and `{% endctrans %}`. The condition is evaluated exactly once. When true, the first branch follows `trans` message, variable, pluralization, trimmed/notrimmed, old/new-style, autoescape, and extraction behavior. When false, the fallback is an ordinary Jinja body and only that body is evaluated; it may contain filters, conditions, includes, and other normal constructs.

An optional `context="literal"` header value selects pgettext or npgettext for the true branch and appears in normal extraction. It may occur once and must be a string literal. The plural selector follows ordinary `trans` rules. A missing fallback, duplicate context, nonliteral context, multiple plural sections, or unclosed block raises the corresponding compile-time template error.

Extraction reports only the true translation message. Fallback text and expressions are not translation messages unless they independently call a configured gettext function.
