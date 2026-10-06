# Reference annotation audit

The audit covers 42 reference norms and 84 task–norm associations: three norms
and six associations per snapshot. One author reviewed these annotations with
model assistance for evidence inspection and preliminary judgments.

| File | Contents |
| --- | --- |
| `sample.json` | Selected norm and association identities and the sampling seed |
| `norms.jsonl` | Norm validity, evidence support, and repository evidence |
| `associations.jsonl` | Applicability, prompt explicitness, and supporting evidence |
| `human_review.csv` | The author's confirmed judgments for all 126 sampled items |
| `protocol.json` | Sample sizes and assessment protocol |

All 42 sampled norms are valid; 41 have full evidential support and one has
partial support. All 84 associations are applicable. Their task descriptions
state the norm explicitly in 11 cases, partly in 20, and omit it in 53.

Review IDs join the confirmation table to the assessment and sample records.
The CSV identifies the reviewer and review date, and records separate judgments
for validity, evidence support, applicability, and prompt explicitness. Fields
that do not apply to an item's kind are empty.

```bash
python tools/validate_release.py
python -m reproduction
```

Validation checks all confirmation identities and judgments against the audit
records. Reproduction verifies the reported counts.
