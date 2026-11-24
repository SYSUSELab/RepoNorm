# Runtime assets

These manifests identify downloadable runtime assets by size and SHA-256.
The assets themselves are stored under ignored output directories.

The frozen Tiktoken image needs its registered encoding files and the NumPy
1.26.4 CPython 3.11 manylinux wheel. Preparation downloads and verifies them,
then evaluation mounts them read-only. NumPy is installed with
`--no-index --no-deps`. The Outlines wheel manifest similarly identifies the
CPU runtime dependencies. Docker evaluation remains offline.

Use `python -m evaluation prepare --task TIK-F05` from the repository root.
A missing or changed asset is an environment preparation error, not a failed
functional test. These assets are dependencies, not repository knowledge or
task labels.
