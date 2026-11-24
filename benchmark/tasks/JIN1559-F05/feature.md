**Title:** Validate variables declared or referenced by `trans` blocks

Translation variable names must be Python identifiers. The names `gettext`, `ngettext`, `pgettext`, `npgettext`, and `_` are reserved and raise `TemplateSyntaxError` when declared in a `trans` header or referenced in its body. This is an intentional compatibility change for templates that previously used one of those names as a formatting placeholder.

An assigned translation expression is valid only when its complete expression tree consists of constants, names, attribute access, filters, tests, or calls whose callee is a simple name. Arguments and keyword values must follow the same rule; dynamic `*args` and `**kwargs` are rejected. Other forms, including item access, arithmetic, conditional expressions, and method calls, raise `TemplateSyntaxError`. Expression errors identify the declared variable and contain `template data` and `translation contexts`; reserved-name errors contain `conflicts with translation function names`.

Validation applies during compilation to singular and plural blocks and works with `trimmed` and `notrimmed`. Existing duplicate-variable and ordinary parser errors remain in effect.
