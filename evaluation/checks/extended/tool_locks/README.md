# Snapshot-specific static tools

`click-2025.txt` records the active Ruff 0.11.13 hook and the mypy 1.16.0 and
Pyright 1.1.401 versions selected for Click snapshot
`5c1239b0116b66492cdd0848144ab3c78a04495a`. It does not lock every transitive
dependency. Preparation records the actual installed tool versions.

Prepare resources from the repository root with
`python -m evaluation prepare --task CLI2956-F01`. The checker uses the target
snapshot's configuration and compares type diagnostics before and after the
patch. `baseline_unchanged` means existing diagnostics were preserved, not that
the upstream project is free of diagnostics.

Functional and behavioral checks run in frozen Docker images without network
access. Static-tool images are separate from the target runtime. The minimum
Python syntax/API checks do not represent a full interpreter or platform matrix.
