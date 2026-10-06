# Literal filename handling for editor commands

Add `escape_shell: bool = False` to `click.edit`. When enabled, treat the filename as one literal editor argument, so spaces, empty strings, quotes, newlines, semicolons, backticks, dollar expansions, and command substitutions cannot alter the command or execute a payload. This applies to direct filenames and temporary filenames.

The configured `editor` remains a trusted editor command and must keep its existing embedded-argument behavior, such as `editor="vim -f"`; this option does not parse untrusted editor configuration. `False` preserves existing behavior. Existing environment, return, and nonzero-exit behavior remain unchanged. The task permits a correctly quoted shell command or a `shell=False` argv implementation and prescribes no private quoting helper.
