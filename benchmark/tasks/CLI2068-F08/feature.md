# Launch the editor in a separate process group

Add `process_group: bool = False` to `click.edit`. When true, create the editor session in a process group or session distinct from the calling process on a platform that supports this. On platforms where the process API has no such capability, continue launching the editor normally. When false, preserve existing process creation exactly.

The option applies to direct-file and temporary-text edits and retains custom environment, return, and nonzero-exit behavior. The public requirement is the observable group separation; implementations may use a new session, a process-group creation flag, a safe adapter, or another equivalent mechanism. In particular, no `preexec_fn` or private helper is required.
