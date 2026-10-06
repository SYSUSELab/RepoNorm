# Set the editor working directory

Add `working_dir: str | os.PathLike[str] | None = None` to `click.edit`. When provided, launch the editor with that directory as its current working directory. A relative direct `filename` is therefore interpreted by the editor relative to `working_dir`; the temporary file created for the `text` form must remain usable from that directory. `working_dir=None` preserves the caller's inherited current directory.

Forward a custom `env` together with the working directory. If process creation rejects a missing or inaccessible directory, preserve Click's existing `ClickException` wrapping and exception chaining. The task does not prescribe whether a PathLike is normalized before passing it to the process API.
