**Title**: Validate nested Context lifetimes through public creation paths

Add `validate_nesting: bool = False` to Click Contexts. Validation is disabled by
default and enabled validation is inherited by descendants.

A child is open after construction with a parent, whether created directly with
`Context(..., parent=...)`, through `Command.make_context(parent=...)`, or during
normal command execution. A validated parent must refuse to close while any
child remains open, raising `RuntimeError` that identifies the outstanding child
command(s). No ordering of multiple names is required. The rejected close must
leave the parent's resources available for subsequent cleanup.

Successfully closing a child removes it from the parent's outstanding children.
Entering a previously closed child opens and tracks it again. Support multiple
children and levels; do not automatically close descendants to hide an invalid
close order. Failed context-manager exit must still restore the current-context
stack, so that closing children and then retrying the parent can recover.
Failure while parsing a newly created child must not leave a phantom open child.

Normal CLI execution (including chained commands) and shell completion must
continue working without spurious nesting errors. Do not globally disable
validation to achieve that compatibility.

