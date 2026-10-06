# Logical template-path normalization

Normalize requested template names in `FileSystemLoader` and `PackageLoader`
before resolving them. The same rules apply on every operating system: both `/`
and `\` are logical separators; empty segments and `.` are removed; an internal
`..` removes the preceding ordinary segment. Thus `foo/../test.html` resolves as
`test.html`, and mixed-separator forms resolve consistently.

Normalization must preserve the loader root boundary. Reject a leading `..`, any
`..` that would pop beyond the logical root, names beginning with `/` or `\`, UNC
forms, and names beginning with an ASCII drive prefix such as `C:`. Rejection
raises `TemplateNotFound` containing the original request name. A normalized name
that is safe but absent also reports the original request name.

Apply the normalized logical name to both directory and archive package lookup.
The existing physical filename and freshness callback behavior are unchanged.
Environment caching remains keyed by the original public request, so two spelling
variants may have separate compiled cache entries. Other loader classes are not
changed by this feature.
