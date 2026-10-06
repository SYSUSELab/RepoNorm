# Human review

`comparison.csv` contains 140 entries assessed by two human reviewers. Each
view contributes 70 entries: five from each of the 14 repository snapshots.

| File | Contents |
| --- | --- |
| `comparison.csv` | Entry identities, model verdicts, and the two human verdicts |
| `selection.json` | Selected claims, snapshot bindings, ranking hashes, and review IDs |
| `protocol.json` | Sampling seed, ranking rules, and snapshot-to-pool mapping |

The CSV fields are:

- `review_id`: the sampled entry identifier.
- `view`: `accepted_only` or `accepted_after_review`.
- `pool`, `snapshot_commit`, `claim_id`: the repository snapshot and claim.
- `historical_model_verdict`: the model verdict in the precision sample.
- `reviewer_a_verdict`, `reviewer_b_verdict`: the two human judgments.

Verdicts are `correct`, `incorrect`, or `undetermined`.

## Agreement

Reviewer agreement counts entries with equal human verdicts: **126/140 = 90.00%**.
Among these agreed entries, the shared verdict matches the model on
**114/126 = 90.48%**. The second rate uses only the entries on which the two
reviewers agree; it does not use all 140 entries as its denominator.

## Reproduce selection and analysis

```bash
python -m reproduction.human_review
python -m reproduction.human_review --output outputs/human-review
```

The first command validates selection and reports total and per-view agreement.
The second also writes selected entries, blinded claim objects, and separately
ordered input lists for reviewers A and B. Use a new output directory. The lists
contain semantic claim fields and evidence pointers without view labels or
judgments. `selection.json` provides the coordinator's identity mapping.

Selection ranks claims within each snapshot and view using the seed and hash
formula in `protocol.json`, then takes the first five. It uses the included
precision samples without consulting model or human verdicts. Running
`python -m reproduction` includes these statistics in `human_review.csv`.
