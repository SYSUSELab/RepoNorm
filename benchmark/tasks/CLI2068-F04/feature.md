# Additional editor arguments

Add an optional `editor_args` iterable of strings to `click.edit`. Insert these arguments, in order, between the configured editor command and the filename for both direct-file and temporary-text edits. Preserve an editor configuration that already contains arguments, such as `editor="vim -f"`. Copy a one-shot iterable once. Reject any non-string item with `TypeError` before launching the editor; `None` and an empty iterable preserve existing behavior.

Each added argument and the filename must retain its exact boundary and literal value, including empty strings, spaces, quotes, newlines, semicolons, backticks, and command substitutions. Existing custom environment, edit result, and nonzero-exit behavior remain unchanged. Safe argv and correctly quoted shell commands are both valid; no quoting helper or private command representation is prescribed.
