**Title**: Add Environment-Conditional Prompt Filters

Add an optional `conditional_filters` mapping to `@outlines.prompt`,
`Prompt.from_str`, and `Prompt.from_file`. Each outer key names a condition and
maps to the Jinja filters installed when that condition is selected.

Selection happens once, when the prompt is constructed. Later environment changes
do not alter an existing prompt. Environment names and values are compared with
ASCII case-insensitive exact equality using these rules:

1. `OUTLINES_DEBUG=true` activates `debug`.
2. `ENVIRONMENT=production` activates `production`.
3. `ENV=testing`, `OUTLINES_TESTING=true`, or `TESTING_MODE=on` activates
   `testing`.

When more than one available condition is active, priority is `debug`, then
`production`, then `testing`. An active standard condition absent from the supplied
mapping is skipped. If none of those conditions both matches and exists, select the
alphabetically first supplied condition key. `None` and an empty mapping install no
conditional filters.

The selected filters apply while the template is compiled, work in included and
inherited file templates, and take precedence over same-name Jinja or Outlines
filters for that prompt. Filters belonging only to unselected conditions are not
installed; referring to one therefore produces Jinja's normal unknown-filter error.
The existing bare decorator remains supported.
