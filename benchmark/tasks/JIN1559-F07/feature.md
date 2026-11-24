# Fallback translation chains

Add an opt-in installer for ordered gettext fallback providers.

`Environment.install_gettext_callables_with_fallback` has this public signature:

```python
install_gettext_callables_with_fallback(
    gettext,
    ngettext,
    fallback_gettext_funcs=None,
    fallback_ngettext_funcs=None,
    newstyle=None,
    pgettext=None,
    npgettext=None,
    fallback_pgettext_funcs=None,
    fallback_npgettext_funcs=None,
)
```

Each fallback argument is an optional finite iterable of callables matching its
primary provider's signature. The installer snapshots each iterable when called.
A missing or empty iterable behaves like `install_gettext_callables`. Supplying a
non-empty contextual fallback iterable without the matching `pgettext` or
`npgettext` primary raises `ValueError` naming the required primary.

Call the primary provider first. A `gettext` or `pgettext` result is missing when
it equals the source message. An `ngettext` or `npgettext` result is missing when
it equals the selected source form: singular for `n == 1`, plural otherwise. Only
on a miss, call matching fallback providers in their declared order and stop at
the first non-missing result. If every provider misses, return the selected source
message. Provider exceptions propagate without calling later providers. Context
providers receive the context string on every attempt.

The gettext and ngettext chains must work for direct calls and for `{% trans %}`
singular and plural blocks in both old-style and new-style gettext modes.
Contextual fallback chains apply to direct `pgettext` and `npgettext` calls; this
feature does not add new template tag syntax. New-style formatting and
autoescaping happen after a provider is selected, using the same behavior as the
existing installer. Calling the installer again replaces all installed providers
and chains. Existing templates that do not use this installer are unchanged.
