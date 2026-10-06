# Search-path priorities

Allow items in a `FileSystemLoader` search-path sequence to be written as
`(path, priority)`. `path` accepts the same string and `os.PathLike` values as an
ordinary item. `priority` must be an integer; booleans are rejected. Ordinary
path items have priority 0.

Sort the complete sequence by descending priority before storing it in the public
`searchpath` list. Python's stable ordering is part of the contract: entries with
the same priority retain their input order. Negative values are valid, duplicate
paths remain duplicate entries, and source lookup and listing use the same sorted
list. A sequence containing only ordinary paths behaves as before.

A tuple item is reserved for the exact `(path, priority)` form. Tuples of another
length, non-integer priorities, booleans, and values that are not valid paths
raise `TypeError` during construction. They are configuration errors and are not
silently skipped or reduced to one tuple element.
