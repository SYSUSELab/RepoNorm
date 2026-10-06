# Sequential option validators

Add an optional `validators` sequence to `click.Option` and the `@click.option` decorator. Each element must be callable and receives `(ctx, param, value)`. After Click has converted the value and run the option's existing callback, call validators in declaration order, passing each return value to the next validator. The final return value becomes the exposed option value. This applies equally to command-line, environment, default, prompt, and `multiple=True` values.

Reject a string or any sequence containing a non-callable at option construction with `TypeError`; `None` and an empty sequence disable the feature. Exceptions retain Click's normal parameter behavior, including `BadParameter` prompt retry and short-circuiting the remaining validators. The task does not prescribe a private storage type or whether the chain is implemented in `process_value` or a composed callback.
