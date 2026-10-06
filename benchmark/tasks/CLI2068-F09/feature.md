# Isolated editor environment

Add `isolate_env: bool = False` to `click.edit`. When true, construct the editor environment from this fixed platform-support allowlist (matched case-insensitively): `PATH`, `HOME`, `USER`, `USERNAME`, `LOGNAME`, `SHELL`, `TERM`, `DISPLAY`, `XAUTHORITY`, `WAYLAND_DISPLAY`, `LANG`, `LANGUAGE`, `LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`, `TMPDIR`, `TMP`, `TEMP`, `SYSTEMROOT`, `COMSPEC`, and `PATHEXT`. Do not inherit any other parent variable. Variable values are copied unchanged.

Apply the caller's explicit `env` mapping afterward, so it can add or override any name, including a credential-like name, by deliberate request. With isolation disabled, preserve Click's existing environment behavior. The option applies to direct and temporary-file edits. This deterministic contract does not classify variables with name substrings and does not prescribe a private filtering helper.
