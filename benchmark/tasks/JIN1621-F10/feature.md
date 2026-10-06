**Title:** Add template path aliases to loaders

Every built-in loader exposes `add_alias(alias, target)` and `remove_alias(alias)`. Alias and target must be strings. An alias is resolved exactly once by the loader that owns it, before that loader performs its normal loading or composite routing; aliases do not chain. Leaf, PrefixLoader, ChoiceLoader, and ModuleLoader support aliases. A missing target raises `TemplateNotFound` for the target name. Removing an unknown alias is harmless, and adding an existing alias overwrites it.

Alias mutation invalidates that alias in every Environment cache which has loaded it, so the next normal request observes the update or removal. `list_templates` includes an alias when its one-step target appears in that loader's ordinary listing; broken and chained aliases are omitted. Direct target entries remain listed. Existing behavior is unchanged until an alias is added.
