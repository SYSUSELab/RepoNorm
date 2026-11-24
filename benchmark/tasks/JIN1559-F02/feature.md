**Title:** Add translation domains to `trans`

A translation block may declare one literal domain in any normal header position: `{% trans domain="admin", count=n %}...{% endtrans %}`. Domain singular and plural messages use `dgettext(domain, message)` and `dngettext(domain, singular, plural, n)`. Without `domain`, existing gettext behavior is unchanged.

`Environment.install_gettext_callables` accepts optional keyword-only-compatible `dgettext` and `dngettext` callables after the existing arguments. If omitted, domain blocks fall back once to the installed ordinary `gettext` or `ngettext`. Reinstalling callables replaces all previous domain behavior. `uninstall_gettext_translations` removes domain globals too.

Domains work with old- and new-style gettext, variables, pluralization, and whitespace modifiers. A duplicate domain raises `TemplateAssertionError` mentioning it was defined twice. A domain value that is not a string literal raises `TemplateAssertionError` mentioning `domain`, `string`, and `literal`.
