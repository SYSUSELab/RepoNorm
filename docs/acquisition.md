# Acquiring repository norms

RepoNorm analyzes a clean Git snapshot with its available history. Its pipeline
performs source discovery, explicit extraction and current-source inference,
evidence verification, counterexample checks, and candidate-conditioned history
inspection. It exports accepted claims as a static view for a coding agent.

Install the core package with the root `requirements.txt`. The core itself
supports Python 3.10+, while the complete experiment environment uses Python
3.12. Keep acquisition and package output outside the target repository.

## Offline smoke example

Create a small disposable repository and run both acquisition stages with the
deterministic fake agent. These commands require no credentials or model calls.
Use a new output directory if repeating the example.

```bash
mkdir -p outputs/smoke/repository
git -C outputs/smoke/repository init -q
printf 'def add(a, b):\n    return a + b\n' > outputs/smoke/repository/example.py
printf '# Example\n\nKeep public functions documented.\n' > outputs/smoke/repository/README.md
git -C outputs/smoke/repository add .
git -C outputs/smoke/repository -c user.name=Smoke -c user.email=smoke@example.invalid -c commit.gpgsign=false commit -qm 'Initial fixture'
repo-memory-v3 discover outputs/smoke/repository \
  --config repo_constraint_memory_v3/configs/hardened.yaml \
  --agent-profile repo_constraint_memory_v3/configs/agents/offline_fake.example.yaml \
  --mock-agent --out outputs/smoke/acquisition
repo-memory-v3 build outputs/smoke/repository \
  --config repo_constraint_memory_v3/configs/hardened.yaml \
  --agent-profile repo_constraint_memory_v3/configs/agents/offline_fake.example.yaml \
  --acquisition outputs/smoke/acquisition --mock --out outputs/smoke/package
repo-memory-v3 validate-unified outputs/smoke/package --repository outputs/smoke/repository
```

The fake discovery agent emits empty proposals. This checks the artifact and
execution contracts, not the scientific quality of generated norms.

## Real acquisition

Real source discovery requires Linux, Git, bubblewrap (`bwrap`), and the
configured mini-SWE-agent 2.4.6 runner:

```bash
python -m pip install 'mini-swe-agent==2.4.6'
repo-memory-v3 discover /path/to/clean/repository \
  --config repo_constraint_memory_v3/configs/hardened.yaml \
  --agent-profile repo_constraint_memory_v3/configs/agents/mini_swe_agent_2_4_6.example.yaml \
  --model-profile deepseek-flash-nonthinking --out outputs/example/acquisition
repo-memory-v3 build /path/to/clean/repository \
  --config repo_constraint_memory_v3/configs/hardened.yaml \
  --agent-profile repo_constraint_memory_v3/configs/agents/mini_swe_agent_2_4_6.example.yaml \
  --acquisition outputs/example/acquisition --out outputs/example/package
```

The model profile specifies its credential environment variable, endpoint, model,
and inference settings. Export the corresponding credential in the host
environment before running. `deepseek-flash-nonthinking` is the recorded
knowledge-generation profile; it differs from the thinking-enabled coding
configuration. Custom execution profiles can be supplied as YAML paths.
The build uses the discovery receipt's bound profile.

For the benchmark, the launcher clones or verifies the exact registered
snapshots and creates separate output directories:

```bash
./run_cooperbench_norm4code.sh --list
./run_cooperbench_norm4code.sh --target tik-4560a8896f5f --clone-only
./run_cooperbench_norm4code.sh --target tik-4560a8896f5f
# All 14 snapshots:
./run_cooperbench_norm4code.sh --all
```

The launcher uses `python3` from the active environment; override it with
`REPONORM_PYTHON` if needed. `--mock` selects the offline profile. Existing
packages are skipped unless `--rerun` requests a fresh directory. Snapshot
preparation does not supply benchmark statements, labels, or reference patches
to acquisition.

## Validate, export, and review

```bash
repo-memory-v3 validate-unified outputs/example/package --repository /path/to/clean/repository
repo-memory-v3 export-consumer outputs/example/package \
  --repository /path/to/clean/repository --profile accepted_only \
  --out outputs/example/consumer
repo-memory-v3 validate-consumer outputs/example/consumer \
  --source-run outputs/example/package --repository /path/to/clean/repository
```

Validation distinguishes a structurally valid partial package from an invalid
package. Preserve completion status and evidence gaps. A consumer should read
`START.md`, select relevant claims, and retain their conditions, exceptions, and
evidence strength. The [consumption skill](../skills/norm4code/SKILL.md) describes
this behavior. Static views do not require the acquisition service at coding time.

Optional `review` starts a local review session without model calls;
`review-publish` publishes the decisions recorded for that session.
`review-auto` uses a separately configured external reviewer and isolation
preflight. Review output remains bound to its source package and snapshot.
Consult each command's `--help` for inputs, and retain whether judgments came
from a human or an automated reviewer.

Run the core test suite from `repo_constraint_memory_v3/` with `python -m pytest`.
The package uses implementation version 0.18.0 and artifact schema
`4.11-partition-only-required-evidence`.
