**Title**: Add Opt-In Option Value Assignment History

Add opt-in option value tracking to Click commands and groups so an application can inspect values that were actually assigned to an option in the current invocation context.

## Public API

Both `Command` and `Group` accept a new keyword argument:

```python
track_changes: bool = False
```

It is available through `@click.command(track_changes=True)` and `@click.group(track_changes=True)`. The default remains disabled and must not add history work during ordinary parsing.

The active `Context` provides:

```python
ctx.get_option_history(name)
```

`name` may be an option's Python parameter name, a declared long option such as `"--count"`, or a declared short option such as `"-c"`.

- When tracking is disabled on the current command, the method returns `None`.
- When tracking is enabled but `name` does not identify an option on the current command, it returns `[]`.
- Positional arguments are not option history entries.
- Each returned item represents a value that was actually assigned to the known option in this `Context`, in assignment order.

Each entry has this shape:

```python
{
    "value": assigned_value,
    "source": ParameterSource.COMMANDLINE,
    "timestamp": 1234567890.123,
}
```

During normal parsing, Click records the final exposed value after type conversion and its callback. `source` is the existing `ParameterSource` member selected by Click, including `COMMANDLINE`, `ENVIRONMENT`, `DEFAULT_MAP`, `DEFAULT`, or `PROMPT`.

If application code later assigns another value through `ctx.params[parameter_name]`, that real assignment is appended with `source=None`; it must not inherit the parser's earlier source. Repeated assignments remain in order, so earlier values that were actually replaced are retained. `timestamp` is a floating-point Unix timestamp recorded for each assignment.

Click must not evaluate an unselected lower-precedence default, environment source, default-map value, or callback merely to manufacture history. Repeated command-line tokens discarded inside parsing are likewise not assignments. The initial parse therefore normally contributes one entry even when lower-precedence candidates existed.

The returned list and entry dictionaries are snapshots. Mutating them, or assigning a later value to the context, must not retroactively change an earlier returned snapshot.

## Behavior and isolation

Tracking observes Click's existing source precedence and conversion pipeline and must not change the selected value. A multiple option is initially recorded once with its final tuple. Each invocation creates a fresh history, and group and subcommand contexts keep independent histories. Enabling tracking on a group does not implicitly enable it on a subcommand, or vice versa.

No private storage name, mapping subclass, lookup helper, or timing implementation is part of the public contract.
