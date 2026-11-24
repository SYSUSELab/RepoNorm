# Opt-in search-path expansion

Add an `expand_searchpath=False` keyword argument to `FileSystemLoader`. The
default preserves every search-path string literally, including `$`, `${...}`,
and `~` characters.

When enabled, convert each string or path-like entry with `os.fspath`, then apply
`os.path.expandvars` followed by `os.path.expanduser` exactly once during loader
initialization. Store and use those expanded paths for all later lookups. Changes
to the process environment after construction do not rewrite the loader's search
paths.

Expansion follows the host operating system. Undefined environment variables
remain in the path as documented by `expandvars`; an empty variable may produce
an absolute or relative path. A single path and a sequence of paths, including
`os.PathLike` values, retain their existing ordering and lookup behavior.
