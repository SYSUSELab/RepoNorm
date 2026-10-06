# RepoNormBench

`manifest.json` is the single task registry: 121 tasks, six repositories, and
14 source snapshots. Task IDs are stable keys shared by every result and norm
annotation. Tasks can be selected by task ID or snapshot ID.

| Repository | Tasks | Snapshots | Reference norms |
| --- | ---: | ---: | ---: |
| Outlines | 22 | 3 | 23 |
| DSPy | 23 | 3 | 24 |
| Tiktoken | 10 | 1 | 6 |
| Click | 27 | 3 | 38 |
| Jinja | 30 | 3 | 59 |
| DirtyEquals | 9 | 1 | 10 |
| Total | 121 | 14 | 160 |

## Task files

Each `tasks/TASK-ID/` directory contains:

- `feature.md`: the coding task statement.
- `feature.patch`: a functional reference implementation.
- `tests.patch`: the hidden functional test patch.
- `norm_reference.patch`: a reference patch used for norm-checker calibration.

The manifest binds these files to SHA-256 hashes, an upstream source commit, a
Docker image ID, the functional test paths, and the checker backend. The
`standard` and `extended` backends preserve the existing task-specific oracle
implementations; they are internal dispatch details of the unified interface.
Fourteen snapshots use 15 distinct images because two DSPy task environments
share a source commit.

`norms.json` contains 160 norms and 1,024 positive applicability associations.
Each norm has an English statement, condition, scope, exceptions, evidence
locations and hashes, and any supporting history. Association records preserve
the applicable task–norm pair and the original prompt-explicitness label; their
`applicability_condition_ref` identifies the norm's condition. Verbose per-pair
annotation narratives are omitted in this compact export. Annotation identity
is recorded as model-assisted source review.

`runtime.json` preserves frozen shim hashes and base/reference functional-test
expectations. Standard preflight requires the recorded base exit with nonzero
test failures and no skips or infrastructure errors; exact base counts are
audited because exception cleanup can vary. Gold counts must match exactly.
The Jinja erratum preserves its original base-fails/gold-passes contract without
a fixed count reference. Source repository sizes and test-change counts are in
[`data/benchmark/`](../data/benchmark/).

## Historical test variant

The default `JIN1559-F07/tests.patch` removes an incidental exact error-message
wording requirement. The earlier patch is preserved as
`JIN1559-F07/variants/message-literal-tests.patch`. DeepSeek and Qwen's selected
executions used that variant; Gemini's used the default. Each execution record
states its `test_variant`. The task statement and reference implementation are
unchanged. Historical outcomes are retained as observed, not rescored and
relabeled against a different test patch.

## Inspect and prepare

```bash
python runner/dataset.py list
python runner/dataset.py verify
python runner/dataset.py clone --task CLI-F01
```

Cloning creates a clean detached checkout with full Git history under
`outputs/repositories/SNAPSHOT-ID`. Existing checkouts must match their expected
origin and commit and be clean; the command does not reset user changes. See
[execution instructions](../docs/experiments.md) for Docker preparation.

Reference patches, hidden tests, norm labels, and evaluation resources are
host-side materials. Coding agents receive the task statement, the unmodified
source snapshot, and the selected static knowledge view.
