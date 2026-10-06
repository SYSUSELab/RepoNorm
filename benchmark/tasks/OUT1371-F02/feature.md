**Title**: Add a Global Prompt Filter Registry

Add `register_filter(name, callable)` and `unregister_filter(name)` to
`outlines.prompts`. A registered filter is available to prompts subsequently
constructed through bare `@outlines.prompt`, `Prompt.from_str`, and
`Prompt.from_file`, including included and inherited file templates.

Registering an existing name replaces its registered callable. Unregistering an
unknown name is a no-op. Registry changes affect only templates constructed after
the change: an existing `Prompt` retains the filter callable captured by its Jinja
environment. A global filter may replace a same-name Jinja default for new prompts,
but the Outlines-specific `name`, `description`, `source`, `signature`, `schema`,
and `args` filters remain protected. The existing APIs behave as before when no
global filters are registered.
