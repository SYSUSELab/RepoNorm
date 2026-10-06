# RepoNorm

RepoNorm acquires repository-specific development norms from a fixed source
snapshot and its Git history, grounds them in repository evidence, and exports
static knowledge for coding agents. This repository contains its implementation,
RepoNormBench, and the selected data needed to inspect and reproduce the reported
experiments.

RepoNormBench is one dataset of **121 tasks** from **6 Python repositories** at
**14 snapshots**, with **160 reference norms** and **1,024 task–norm associations**.
The result release contains **1,089 executions**: three coding models, three
conditions, and every benchmark task.

## Reproduce the reported results

Use Python **3.12**, Git, and a virtual environment. Run these commands from the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/validate_release.py
python -m reproduction
```

This path uses the included data and requires no API credentials, model calls,
or Docker. It recomputes the main results, paired
confidence intervals, significance tests, sensitivity analyses, quality
summaries, calibration counts, and recorded costs. Tables and figures are written
to `outputs/reproduction/`. For a quick check of point estimates, use
`python -m reproduction --skip-bootstrap`.

See the [reproduction guide](docs/reproduction.md) for metrics, output files,
and expected counts; the [data guide](data/README.md) describes every released
input and its provenance.

## Run RepoNorm or new experiments

- [Acquire norms](docs/acquisition.md): analyze a repository, validate a package,
  and export an agent-readable view; includes an offline smoke example.
- [Run coding and scoring experiments](docs/experiments.md): prepare frozen
  snapshots and Docker resources, use the released knowledge, generate a
  CodeWiki baseline, and score new patches.
- [Inspect RepoNormBench](benchmark/README.md): task artifacts, snapshot bindings,
  reference annotations, and the preserved test variant.

The public conditions are `raw`, `codewiki`, and `reponorm`. The coding models are
DeepSeek V4 Flash, Qwen3.5 Flash, and Gemini 3.1 Flash Lite. Exact model aliases,
provider routes, budgets, and settings are retained with each selected execution;
new calls to a hosted model are not deterministic replays of the saved results.

## Repository layout

| Path | Contents |
| --- | --- |
| `repo_constraint_memory_v3/` | Acquisition, validation, export, and review implementation |
| `skills/norm4code/` | Static knowledge consumption instructions |
| `benchmark/` | Unified task index, task artifacts, norms, and runtime contracts |
| `data/` | Selected results, patches, knowledge packages, and assessment data |
| `reproduction/` | Analysis of the released data |
| `runner/` | Snapshot preparation and isolated coding execution |
| `evaluation/` | Norm checkers and patch scoring |
| `CooperBench/`, `CodeWiki/` | Adapted upstream components |
| `tests/` | Runner and integration tests |

## Release scope

The release includes 14 Accepted-Only knowledge views (1,990 entries), 14
Accepted-after-Review views (6,277 entries), and 14 CodeWiki views. It also
includes precision samples, coverage assessments, and 140 entries with judgments
from two human reviewers, together with sample selection and agreement analysis.

[Third-party provenance](docs/third-party.md) identifies the included components.
