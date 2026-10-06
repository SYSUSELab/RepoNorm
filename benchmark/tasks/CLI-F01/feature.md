**Title**: Close resources created during shell completion

Shell completion can leave resources opened by file-valued options unclosed,
which later produces `ResourceWarning`. Fix completion context lifetime
management so that resources created while resolving a completion request are
closed before the request finishes.

The fix must cover top-level commands, nested subcommands, and chained command
groups without changing the completion candidates or normal command execution.
Running completion for a command containing a `click.File` option must not
leave an unclosed file warning after garbage collection.

Resources must remain available while producing the completion candidates, then
be released even if parsing or the completion callback raises. Do not swallow
those exceptions. Repeated requests must not retain resources from an earlier
request.
