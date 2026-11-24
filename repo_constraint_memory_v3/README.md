# RepoNorm acquisition implementation

This package acquires repository-specific norms from a fixed Python repository
snapshot. It combines explicit extraction and current-source inference with
evidence grounding, counterexample checks, and candidate-conditioned Git history.
It publishes static norm views for existing coding agents.

Requires Python 3.10 or newer. From the repository root:

```sh
python -m pip install -e './repo_constraint_memory_v3[test]'
repo-memory-v3 --help
```

The implementation version is 0.18.0 and the artifact schema is
`4.11-partition-only-required-evidence`. The package and command names remain
stable for configuration and artifact compatibility.

See the root [README](../README.md) for installation and reproduction entry points.
The [acquisition guide](../docs/acquisition.md) describes model selection, discovery,
building, validation, and the distinction between partial and invalid packages.

Run the core tests from this directory:

```sh
python -m pytest
```
