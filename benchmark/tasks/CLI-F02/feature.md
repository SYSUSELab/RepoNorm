**Title**: Add an exit callback to Click contexts

Allow `on_exit=callback` when creating a command context. The callback receives
the Context being closed and can observe its final command identity and parsed
parameters. Omitting it preserves existing behavior.

The callback runs once per completed teardown, not once per nested entry of the
same Context. Exiting an inner `with ctx` while an outer entry remains open must
not invoke it. A later, separate enter/exit cycle invokes it again. Calling
`ctx.exit()` and subsequently leaving its enclosing context must not invoke it
twice for the same teardown. `scope(cleanup=False)` does not trigger it.

Parent and child contexts may have independent callbacks. Normal commands,
commands that raise, and actual chained subcommands must obey the same rule.
Completion's internal context resolution must not invoke the user callback.

Ignore ordinary `Exception` instances raised by the callback and continue normal
resource cleanup. Do not swallow `KeyboardInterrupt` or `SystemExit`, but still
release resources and restore the current-context stack when they propagate.

