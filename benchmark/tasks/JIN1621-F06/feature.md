# Template path transformation

Add an optional `path_transform` keyword argument to `FileSystemLoader`,
`PackageLoader`, `DictLoader`, `FunctionLoader`, `PrefixLoader`, `ChoiceLoader`,
and `ModuleLoader`. It is a callable that receives exactly one requested template
name and must return a string. `None`, the default, preserves existing behavior.
Support for the source-less `ModuleLoader` is included so aliases can target
precompiled templates as well as source templates.

A loader with a transform calls it exactly once for each uncached public load or
`get_source` operation, before that loader resolves the name. The transformed
name is used for filesystem safety checks, lookup, prefix routing, child
selection, or precompiled module-key lookup. `PrefixLoader` transforms the
complete name before splitting its prefix. `ChoiceLoader` transforms the complete
name before trying children. If a selected child has its own transform, that
child applies its transform once to the name it receives; configuring two loader
layers therefore invokes each configured transform once.

Exceptions raised by a transform propagate unchanged. A non-string return raises
`TypeError` with a message that says `path_transform must return a string`. Empty
input is passed to the callable and may be rewritten. When lookup of a transformed
name fails, `TemplateNotFound.name` is the original name passed to that loader.
Filesystem transforms do not bypass the existing template path safety checks.

Environment template caching remains keyed by the original public request name.
`list_templates()` continues to list discoverable stored names and does not try
to invert or apply a transform. Existing loaders constructed without a transform
retain their behavior.
