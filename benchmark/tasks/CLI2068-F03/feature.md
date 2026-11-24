# Automatic backups for `click.edit`

Add an opt-in `auto_backup: bool = False` keyword to `click.edit`. When a caller supplies an existing direct `filename` and enables the option, copy the file immediately before launching the editor. Temporary files created for the `text` form and direct filenames that do not yet exist do not create backups.

Backup names begin with `{filename}.backup.YYYYMMDD_HHMMSS`. Repeated edits must never overwrite an earlier backup, including calls in the same second; an implementation may add a collision suffix. Preserve the source file metadata, including its permission mode. If backup creation raises `OSError`, raise `ClickException` with a message beginning `Failed to create backup file` and chain the original error. Do not launch the editor after that failure.

The default behavior and existing edit result/error behavior remain unchanged. The task does not prescribe a copy helper or internal method layout.
