# Running new coding and scoring experiments

Run host commands from the repository root on Linux with Git, Docker, and
Python 3.12. Install `requirements.txt` and the selected CooperBench runtime:

```bash
source .venv/bin/activate
python -m pip install -e ./CooperBench
python runner/dataset.py verify
```

Preparation may download upstream repositories, Docker images, encodings, and
wheels. Coding calls use the chosen provider; functional and norm evaluation run
without network access. Outputs are written below `outputs/`.

## Prepare one task

```bash
python runner/dataset.py clone --task CLI-F01
python runner/environment.py prepare --task CLI-F01
python runner/environment.py preflight --task CLI-F01
```

The preflight runs the functional test patch against both the original snapshot
and the functional reference patch, comparing the observed result with recorded
expectations. The default test patch is used. Image IDs, commits, runner/shim
hashes, and runtime assets are checked before execution. `--registry` accepts an
optional Docker Hub mirror prefix; the expected image ID must still match.
Repeat `--task` to select several tasks, or omit it to select all 121.

These images can be large. Outlines runtime preparation includes CPU dependency
wheels; Tiktoken requires frozen encoding files and its NumPy wheel. Preparation
fails on missing or mismatching resources.

## Plan or run coding

An offline plan selects the recorded configuration and stages the released
snapshot-bound knowledge without model calls or Docker execution:

```bash
python -m runner plan --model qwen --condition reponorm --task CLI-F01 \
  --output outputs/qwen-plan
```

A live run requires the credential named by its configuration:

| Model selector | Credential variable | Recorded route |
| --- | --- | --- |
| `deepseek` | `OPENROUTER_API_KEY` | DeepSeek V4 Flash, fixed GMICloud FP8 route, high reasoning |
| `qwen` | `DASHSCOPE_API_KEY` | Qwen3.5 Flash, thinking enabled with an 8,192-token budget |
| `gemini` | `APIYI_GEMINI31_API_KEY` | Gemini 3.1 Flash Lite, medium reasoning |

Set credentials in the host environment, then run:

```bash
python -m runner run --model qwen --condition reponorm --task CLI-F01 \
  --workers 1 --output outputs/qwen-run
```

Choose `raw`, `codewiki`, or `reponorm`; repeat `--condition` or omit it for all
three. Use a new output directory for each run. `--max-calls` explicitly overrides
the recorded logical call budget for a pilot and is recorded as an override.
The default budget is 300 logical calls, with a separate safety step limit.
Per-job `spec.json`, `result.json`, and `worker.log` retain execution details.
`summary.json` reports worker exits and outcomes; execution or isolation errors
produce a nonzero batch exit.
Historical infrastructure-retry settings remain in the source configuration;
the public batch command executes each selected job once and reports failures.
Retrying a job is an explicit new run.

The worker uses the adapted mini-SWE-agent v2 runtime and applies the selected
API retry, cache, compaction, telemetry, and workspace-recovery settings. It
stages repository history and mounts static Markdown knowledge read-only.
The coding container cannot see hidden test patches, reference patches, or norm
labels. Submitted test changes are handled separately from hidden evaluation.

New coding runs use the corrected default tests for every model and condition,
including `JIN1559-F07`; they do not inherit historical test variants from saved
executions. The earlier `message-literal` variant and the paper's recorded results
remain available for historical reproduction. The HF leaderboard uses corrected
evaluation results and is separate from that unchanged paper archive.
New results are new observations: hosted model aliases
and provider backends cannot guarantee identical weights or responses.

## Score submitted patches

Norm-checker resources have their own preparation command:

```bash
python -m evaluation prepare --task CLI-F01
python -m evaluation score --task CLI-F01 --model qwen --condition reponorm \
  --output outputs/qwen-saved-score
python -m evaluation score --task CLI-F01 --patch /path/to/submission.patch \
  --output outputs/new-score
```

Omit `--task` to score all saved patches for a model and condition. Use `--mock`
for an offline wiring check; mock verdicts remain `not_run`. The scorer dispatches
the appropriate internal backend and exports the final `coarse_auto` decisions.
Authored-test delivery follows the reported green-only delivery rule; stricter
diagnostic oracle outputs are not substituted for the published profile.

Runtime and static checks use separate images. Static tool specifications retain
the recorded principal versions; some older profiles do not include complete
transitive dependency locks. A newly prepared image therefore records its actual
installed versions. The included decision tables remain the authoritative
historical results, and recomputing their statistics requires no environment
reconstruction.

## Generate the CodeWiki baseline

The released CodeWiki archives can be consumed directly. To regenerate them,
use a separate environment so its recorded dependencies do not replace the
coding runtime's dependencies. Node.js and bubblewrap are also required.

```bash
python3.12 -m venv .venv-codewiki
.venv-codewiki/bin/python -m pip install -c CodeWiki/constraints.txt \
  -e ./CodeWiki -e ./repo_constraint_memory_v3
PYTHONPATH=CodeWiki .venv-codewiki/bin/python -m integration.run --list
PYTHONPATH=CodeWiki .venv-codewiki/bin/python -m integration.run \
  --target tik-4560a8896f5f --clone-only
```

Prepare the `cl100k_base` tokenizer cache in that environment before generation:

```bash
.venv-codewiki/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
PYTHONPATH=CodeWiki .venv-codewiki/bin/python -m integration.run \
  --target tik-4560a8896f5f --mode probe --mock
PYTHONPATH=CodeWiki .venv-codewiki/bin/python -m integration.run \
  --target tik-4560a8896f5f --mode generate \
  --model-profile deepseek-flash-nonthinking
```

The mock probe makes no provider calls. Real generation requires the profile's
credential. `--mode check` verifies prerequisites, and `--all` selects all 14
snapshots. The adapter records source identity, usage, completion, and environment
versions, and verifies that the source checkout is unchanged. Generated runs
appear in `outputs/codewiki/`; export a selected completed run with
`--export-consumer PATH`. `--allow-partial` explicitly permits export of a partial
run while retaining its status.

## Tests

The runner tests require the CooperBench runtime installed above. The core tests
need only the root requirements; CodeWiki tests use its separate environment.

```bash
python -m pytest -q
(cd repo_constraint_memory_v3 && python -m pytest -q)
.venv-codewiki/bin/python -m pip install -c CodeWiki/constraints.txt pytest pytest-asyncio
PYTHONPATH=CodeWiki .venv-codewiki/bin/python -m pytest -o addopts= CodeWiki/tests
```

The statistical tables can be reproduced from the included data without model
calls or Docker execution; see the [reproduction guide](reproduction.md).
