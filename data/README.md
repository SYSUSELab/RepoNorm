# Experiment data

Stable task, snapshot, model, and condition keys link the experiment data to
[`benchmark/manifest.json`](../benchmark/manifest.json).

## Inventory

| Directory or file | Contents |
| --- | --- |
| `results/executions.jsonl` | 1,089 coding receipts with outcomes, configurations, patches, usage, timing, and test variants |
| `results/configurations.json` | Model and execution settings keyed by configuration ID |
| `results/patches/` | Available submitted patches |
| `results/norm_decisions_*.csv` | 3,072 task–condition–norm decisions per model, using `coarse_auto` |
| `knowledge/` | 14 Accepted-Only archives (1,990 entries), 14 Accepted-after-Review archives (6,277 entries), 14 CodeWiki archives, and file checksums |
| `quality/` | 300 Accepted-Only and 600 Accepted-after-Review precision samples; 160 reference-coverage assessments per view; sampling protocols |
| `human_review/` | 140 entries with two human reviewers' judgments, sample selection, and sampling protocol |
| `reference_audit/` | Selection, evidence, judgments, and author confirmations for 42 norms and 84 associations |
| `calibration/` | 455 patches, expected labels, and 1,351 checker decisions |
| `sensitivity/` | Prompt-revision labels and task statements; the fixed 59 documentation exclusions |
| `benchmark/` | Snapshot size statistics and per-task test construction changes |
| `cost/acquisition.json` | Usage, discovery/build duration, completion state, and delivered counts for 14 snapshots |
| `expected.json` | Exact fractions, interval endpoints, comparison targets, and cost precision |
| `provenance/` | Artifact digests and file checksums |

## Result semantics

Each of the nine model–condition combinations contains 121 executions. The
conditions are `raw`, `codewiki`, and `reponorm`; the latter consumes the
Accepted-Only view. Missing patches and unavailable evidence retain their actual
states. An unavailable patch is not a successful empty change. Norm decisions
include `not_applicable` and errors as well as `pass` and `fail`.

Token fields include completeness metadata and the number of responses with a
reported value. Sums with partial coverage are observed totals. Times describe
agent execution or acquisition stages. Configuration records specify retry,
cache, workspace recovery, concurrency, and provider settings.

Patches retain their submitted bytes. `source_patch_sha256` checks a submitted
patch; `source_receipt_sha256` identifies its execution receipt. Each execution
binds its functional test variant; see the
[benchmark guide](../benchmark/README.md).

## Knowledge and assessments

The knowledge manifest binds each archive to its repository commit, generation
profile, completion state, and checksums. RepoNorm archives contain static
knowledge views with claim details and navigational indexes. CodeWiki archives
contain generated Markdown for the `docs/codewiki/` mount. Partial acquisition
packages retain their `partial` status. Static knowledge archives are inputs for
coding agents; full acquisition-package validation requires an acquisition run.

Precision records contain sampled claims, verdicts, issue tags, and evidence
locations. Coverage records contain matched claims and assessments. The quality
protocols specify the assessment model and sampling rules. The
[reference audit](reference_audit/README.md) contains norm validity, evidence
support, task applicability, and prompt-explicitness judgments, together with
126 author confirmations from a review with model assistance.

The [human review data](human_review/README.md) contain five entries per snapshot
and view, for 140 entries in total. The two human reviewers agree on 126 entries
(90.00%); their shared judgments match the model on 114 of those entries (90.48%).
Sample selection can be reproduced from the included precision samples.

The two RepoNorm knowledge views are distinguished by the manifest's `view`
field: `accepted_only` and `accepted_after_review`. The latter archives are in
`knowledge/accepted_after_review/`. All 6,277 entries are readable, including 146
entries marked `reference_only`; these provide context rather than default
actions. The coding experiments use the Accepted-Only view.

## Integrity and reproduction

Artifact digests are recorded in `provenance/source_artifacts.json`; distributed
files have checksums in `provenance/checksums.json`.

```bash
python tools/validate_release.py
python -m reproduction
```

Validation checks hashes, archive members, task and snapshot bindings, patch
bindings, assessment counts, and sample selection. The analysis recomputes the
included summaries; see the [reproduction guide](../docs/reproduction.md).
