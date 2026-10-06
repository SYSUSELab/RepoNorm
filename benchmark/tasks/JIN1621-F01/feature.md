# Fallback filesystem search paths

Add an optional `fallback_searchpath` keyword argument to `FileSystemLoader`.
It accepts the same single string, `os.PathLike`, or ordered sequence forms as
`searchpath`; `None` and an empty sequence mean no fallback paths. The normalized
public `fallback_searchpath` attribute is a list of filesystem strings.

For every source lookup, search all entries in `searchpath` in their existing
order, then all entries in `fallback_searchpath` in their declared order. Return
the first matching file. This is a named configuration convenience equivalent to
one combined ordered path list. If a primary file that supplied a cached template
is deleted, its `uptodate` callback becomes false and a later reload may resolve
the same name from a fallback path.

`list_templates()` walks primary and fallback directories, removes duplicate
logical names, and returns one sorted list. Existing encoding, link-following,
path-safety, and `TemplateNotFound` behavior remain unchanged. Omitting the new
argument preserves existing behavior.
