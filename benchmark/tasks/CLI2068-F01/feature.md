# Edit multiple files in one editor session

Extend the direct `filename` form of `click.edit` to accept either one string or an iterable of strings. Launch one editor session and pass the filenames in iteration order as distinct literal arguments. A string remains one filename rather than an iterable of characters. Consume a one-shot iterable once. Reject an empty iterable with `ValueError` and a non-string item with `TypeError` before launching the editor.

Filenames containing spaces, quotes, newlines, semicolons, backticks, or command substitutions must retain their exact argument boundaries and must not be executed as shell syntax. Preserve an editor configuration that already contains arguments, custom environment behavior, direct single-file behavior, temporary-text behavior, and existing nonzero-exit errors. The task does not require renaming a private method or using a shell command; safe argv and correct shell quoting are both valid.
