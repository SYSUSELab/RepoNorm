# Editor exit-code handlers

Add an optional `handle_exit_codes` mapping to `click.edit`. Its integer keys are editor exit codes and its values are callables that receive the matching integer code. After the editor exits, invoke the matching handler exactly once for both zero and nonzero codes. Ignore the handler return value. A matched nonzero code is handled and does not raise the usual editing failure; an unmatched nonzero code keeps Click's existing `Editing failed` behavior.

If a handler raises, raise `ClickException` describing the handler failure and chain the original exception. The behavior applies to both direct `filename` edits and temporary-file `text` edits. With no mapping or an empty mapping, all existing behavior and return values remain unchanged. No private storage shape or subprocess construction is required.
