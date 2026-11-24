**Title**: Add transactional Context parsing-state snapshot and restore

Add `Context.save_state()` and `Context.restore_state(snapshot)` for temporarily
changing parsing and presentation state, then restoring it.

The snapshot is a dictionary covering `params`, `args`, protected arguments,
option prefixes, `obj`, metadata, `default_map`, `invoked_subcommand`, terminal
and content widths, parsing flags, `help_option_names`, `token_normalize_func`,
`auto_envvar_prefix`, `color`, `show_default`, and parameter-source tracking.
Use the corresponding Context attribute names as dictionary keys, including
`_protected_args`, `_opt_prefixes`, `_meta`, and `_parameter_source`.

Mutable values must be independent of the source Context. Within a snapshot,
preserve shared references across supported fields and cycles. Restore must
likewise preserve that graph without sharing mutable values with the snapshot.
Snapshots are reusable and multiple snapshots remain independently restorable.

Restoring metadata changes its contents while preserving the existing metadata
dictionary shared by parent and child Contexts. References to that dictionary
within restored state must still point to the shared dictionary.

Restore is transactional: require a dictionary with all supported keys and
dictionary-valued metadata. Missing keys, invalid metadata, or copying failures
must raise without partially changing the Context. Values that cannot be copied
(such as some open resources) may fail; they need not be supported.

Unknown snapshot keys are ignored. Resources, `_exit_stack`, close callbacks,
`parent`, `command`, and arbitrary dynamically added attributes are outside the
contract and must remain unchanged. Contexts using resilient parsing must also
work; changing the shell-completion algorithm is not required.

