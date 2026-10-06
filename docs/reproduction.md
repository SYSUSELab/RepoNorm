# Reproducing the reported analyses

Install the root `requirements.txt` in Python 3.12 and run:

```bash
python tools/validate_release.py
python -m reproduction --output outputs/reproduction
```

The analysis uses only `benchmark/` and `data/`. It does not execute submitted
patches or contact model providers. Its assertions compare recomputed quantities
with `data/expected.json`; a mismatch exits with an error. Expected values are
verification targets, not substitutes for computation.

## Outputs and definitions

| Output | Interpretation |
| --- | --- |
| `ncr.csv` | 72 task-averaged NCR cells across main and sensitivity selections |
| `paired_ncr.csv` | 48 paired NCR differences, with 95% BCa intervals in a full run |
| `functional.csv` | Functional and joint success counts for each model and condition |
| `paired_binary.csv` | 12 exact McNemar comparisons with Holm correction |
| `quality.csv` | Precision and reference-norm coverage for the two knowledge views |
| `human_review.csv` | Agreement between human reviewers and agreement of their shared judgments with the model |
| `cost.csv` | Recorded acquisition and coding token counts and elapsed time |
| `paired_ncr.pdf`, `paired_ncr.png` | Main paired NCR differences and intervals |
| `verification.json` | Verified dataset, calibration, audit, construction, and timing counts |

For each task, NCR is the fraction of applicable, decided norms marked `pass`.
The reported NCR is the mean of those task fractions, giving each eligible task
equal weight. Norms marked `not_applicable` and unavailable decisions do not enter
that fraction. Tasks with no decided norms do not enter its mean. Functional
success retains all 121 tasks per model and condition in its denominator. Joint
success requires functional success and every applicable norm passing.

The scopes are Overall, Contribution (authored tests, documentation, changelog,
inline change notes, style, and typing), Behavioral (the remaining five
categories), and Prompt-omitted. The last scope uses the fixed `omitted` label;
other annotation labels are not silently reclassified as omitted.

Paired differences compare RepoNorm with Raw or CodeWiki on the common eligible
tasks. Each interval uses 100,000 paired BCa bootstrap resamples at 95% confidence.
The preserved task order, seeded PCG64 generator, NumPy 2.0.1, and SciPy 1.16.0
reproduce the recorded intervals. Exact McNemar tests use discordant binary
outcomes and apply Holm correction jointly to the 12 comparisons.

## Sensitivity analyses and supporting counts

The 670 Prompt-omitted associations consist of 651 already absent in the original
statements, 16 with a removed partial reminder, and 3 excluded from the
already-absent subset. Removing tasks with reminder removal leaves 107 tasks.
The documentation sensitivity analysis excludes the same fixed 59 associations
from every condition.

The supporting checks verify:

- 455 calibration patches and 1,351 norm checks: 1,010 expected-compliant checks
  pass, 302 expected-violating checks fail, and 39 expected-violating checks pass.
- A reference audit of 42 norms and 84 associations: 41 fully supported norms and
  one partially supported norm; prompt labels 11 explicit, 20 partial, 53 omitted.
- Precision samples of 300 and 600 claims, with 258 and 451 correct judgments.
- Coverage over 160 reference norms: 35/63 fully/fully-or-partially covered for
  Accepted-Only and 84/125 for Accepted-after-Review.
- Human review of 140 entries: 126 matching reviewer judgments (90.00%), of
  which 114 match the model (90.48%).
- Benchmark test changes: 501 added identities, 503 removed identities, and 78
  retained identities with modified ASTs.

## Human review sample

The human review uses five entries per snapshot and view, selected from the
included precision samples by a seeded hash ranking. Recreate the sample and
blinded reviewer input lists with:

```bash
python -m reproduction.human_review --output outputs/human-review
```

The output directory must be new. The command verifies all 140 selections and
computes agreement from the recorded CSV; it does not perform new reviews.
See the [human review guide](../data/human_review/README.md) for data fields and
denominators.

Token and timing totals come from recorded receipts. Partial usage remains
partial and is not presented as complete provider billing. Acquisition is a
shared cost for 14 snapshots; it is not charged again for every coding model.

`--skip-bootstrap` verifies the point estimates and other summaries without
resampling or emitting the interval figure. It does not verify interval endpoints.
To rerun the actual checkers on saved patches, follow the
[experiment guide](experiments.md).
