#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_DATA_ROOT="$PROJECT_ROOT/outputs"
CONFIG="$PROJECT_ROOT/repo_constraint_memory_v3/configs/hardened.yaml"
CONFIG_EXPLICIT=false
CONFIG_SUFFIX=""
REAL_AGENT_PROFILE="$PROJECT_ROOT/repo_constraint_memory_v3/configs/agents/mini_swe_agent_2_4_6.example.yaml"
MOCK_AGENT_PROFILE="$PROJECT_ROOT/repo_constraint_memory_v3/configs/agents/offline_fake.example.yaml"
DEFAULT_MODEL_PROFILE="deepseek-flash-nonthinking"
WORKFLOW_ID="norm4code-0.18.0-schema-4.11"

declare -a PYTHON_SNAPSHOTS=()
declare -A SNAPSHOT_URL=()
declare -A SNAPSHOT_COMMIT=()

register_snapshot() {
    local id="$1"
    local url="$2"
    local commit="$3"
    PYTHON_SNAPSHOTS+=("$id")
    SNAPSHOT_URL["$id"]="$url"
    SNAPSHOT_COMMIT["$id"]="$commit"
}

# Load the single RepoNormBench snapshot registry.
PYTHON_BIN="${REPONORM_PYTHON:-python3}"
while IFS=$'\t' read -r id url commit; do
    register_snapshot "$id" "$url" "$commit"
done < <("$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(s["id"],s["repository_url"],s["commit"],sep="\t") for s in d["snapshots"]]' "$PROJECT_ROOT/benchmark/manifest.json")
DEFAULT_TARGETS=("tik-4560a8896f5f")

DATA_ROOT="$DEFAULT_DATA_ROOT"
MODEL_PROFILE="$DEFAULT_MODEL_PROFILE"
MODEL_PROFILE_EXPLICIT=false
CLONE_ONLY=false
RERUN=false
MOCK=false
LIST_ONLY=false
ALL_PYTHON=false
declare -a REQUESTED_TARGETS=()

usage() {
    cat <<'EOF'
Usage: ./run_cooperbench_norm4code.sh [OPTIONS]

Clone exact CooperBench snapshots with full Git history and build a fresh,
snapshot-bound RepoNorm package. The default target is the Tiktoken snapshot.

Options:
  --target ID          Select one registered snapshot; repeat for a batch.
  --config PATH        Config YAML; explicit configs use a content-hash run namespace.
  --all                Select all 14 benchmark repository snapshots.
  --clone-only         Clone/verify snapshots without running Norm4Code.
  --rerun              Build a new uniquely named run even if one already exists.
  --mock               Use the provider-free fake acquisition and mock LLM.
  --model-profile ID   Real execution-model profile (default: deepseek-flash-nonthinking).
  --data-root PATH     Shared data directory (default: ./outputs).
  --list               List registered snapshots and exit.
  -h, --help           Show this help.

Without --rerun, a previously published package for the same target and
real/mock profile is reported and skipped. No existing clone or run is replaced.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --target)
            (($# >= 2)) || die "--target requires an ID"
            REQUESTED_TARGETS+=("$2")
            shift 2
            ;;
        --config)
            (($# >= 2)) || die "--config requires a path"
            CONFIG="$2"
            CONFIG_EXPLICIT=true
            shift 2
            ;;
        --all)
            ALL_PYTHON=true
            shift
            ;;
        --clone-only)
            CLONE_ONLY=true
            shift
            ;;
        --rerun)
            RERUN=true
            shift
            ;;
        --mock)
            MOCK=true
            shift
            ;;
        --model-profile)
            (($# >= 2)) || die "--model-profile requires an ID or path"
            MODEL_PROFILE="$2"
            MODEL_PROFILE_EXPLICIT=true
            shift 2
            ;;
        --data-root)
            (($# >= 2)) || die "--data-root requires a path"
            DATA_ROOT="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ "$LIST_ONLY" == true ]]; then
    for id in "${PYTHON_SNAPSHOTS[@]}"; do
        marker=""
        if [[ "$id" == "${DEFAULT_TARGETS[0]}" ]]; then
            marker=" (default)"
        fi
        printf '%s%s\n  repository: %s\n  commit:     %s\n' \
            "$id" "$marker" "${SNAPSHOT_URL[$id]}" "${SNAPSHOT_COMMIT[$id]}"
    done
    exit 0
fi

selection_count=0
[[ "$ALL_PYTHON" == true ]] && ((selection_count += 1))
((${#REQUESTED_TARGETS[@]} > 0)) && ((selection_count += 1))
if ((selection_count > 1)); then
    die "choose either --all or explicit --target values"
fi
if [[ "$MOCK" == true && "$MODEL_PROFILE_EXPLICIT" == true ]]; then
    die "--mock selects the offline test profile; omit --model-profile"
fi

[[ -f "$CONFIG" ]] || die "Norm4Code config is missing: $CONFIG"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd -P)/$(basename "$CONFIG")"
if [[ "$CONFIG_EXPLICIT" == true ]]; then
    CONFIG_SUFFIX="-config-$(sha256sum "$CONFIG" | cut -d ' ' -f 1)"
fi

if [[ "$MODEL_PROFILE" == */* || "$MODEL_PROFILE" == *.yaml || "$MODEL_PROFILE" == *.yml ]]; then
    [[ -f "$MODEL_PROFILE" ]] || die "execution-model profile file does not exist: $MODEL_PROFILE"
    MODEL_PROFILE="$(cd "$(dirname "$MODEL_PROFILE")" && pwd -P)/$(basename "$MODEL_PROFILE")"
fi

if [[ "$ALL_PYTHON" == true ]]; then
    TARGETS=("${PYTHON_SNAPSHOTS[@]}")
elif ((${#REQUESTED_TARGETS[@]} > 0)); then
    TARGETS=("${REQUESTED_TARGETS[@]}")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

for id in "${TARGETS[@]}"; do
    [[ -n "${SNAPSHOT_URL[$id]:-}" ]] || die "unknown snapshot ID: $id (use --list)"
done

[[ ! -L "$DATA_ROOT" ]] || die "data root must not be a symlink: $DATA_ROOT"
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd -P)"
REPOSITORIES_DIR="$DATA_ROOT/repositories"
RUNS_DIR="$DATA_ROOT/norm4code_runs"
CACHES_DIR="$RUNS_DIR/caches"
mkdir -p "$REPOSITORIES_DIR" "$RUNS_DIR" "$CACHES_DIR"
for managed_directory in "$REPOSITORIES_DIR" "$RUNS_DIR" "$CACHES_DIR"; do
    [[ ! -L "$managed_directory" ]] || die "managed data directory must not be a symlink: $managed_directory"
done
cd "$PROJECT_ROOT"

normalize_url() {
    local value="$1"
    value="${value%/}"
    value="${value%.git}"
    printf '%s\n' "$value"
}

ensure_snapshot_clone() {
    local id="$1"
    local url="${SNAPSHOT_URL[$id]}"
    local commit="${SNAPSHOT_COMMIT[$id]}"
    local repository="$REPOSITORIES_DIR/$id"
    local repository_parent
    repository_parent="$(dirname "$repository")"
    mkdir -p "$repository_parent"

    if [[ -L "$repository" ]]; then
        die "repository path must not be a symlink: $repository"
    fi

    if [[ -e "$repository" ]]; then
        [[ -d "$repository/.git" ]] || die "existing path is not an independent Git clone: $repository"
        local actual_url
        actual_url="$(git -C "$repository" remote get-url origin)" \
            || die "existing clone has no origin: $repository"
        [[ "$(normalize_url "$actual_url")" == "$(normalize_url "$url")" ]] \
            || die "existing clone has the wrong origin: $repository"

        local promisor
        promisor="$(git -C "$repository" config --bool --get remote.origin.promisor 2>/dev/null || true)"
        [[ "$promisor" != "true" ]] \
            || die "existing clone is partial/promisor, not a self-contained full clone: $repository"

        local shallow
        shallow="$(git -C "$repository" rev-parse --is-shallow-repository)"
        if [[ "$shallow" == "true" ]]; then
            printf '[deepen] %s is shallow; fetching complete history\n' "$id" >&2
            git -C "$repository" fetch --unshallow --tags origin
            [[ "$(git -C "$repository" rev-parse --is-shallow-repository)" == "false" ]] \
                || die "failed to deepen clone: $repository"
        fi

        if ! git -C "$repository" cat-file -e "${commit}^{commit}" 2>/dev/null; then
            printf '[fetch] %s does not contain its frozen commit; fetching all branches and tags\n' "$id" >&2
            git -C "$repository" fetch --tags origin '+refs/heads/*:refs/remotes/origin/*'
        fi
        git -C "$repository" cat-file -e "${commit}^{commit}" 2>/dev/null \
            || die "frozen commit is unavailable after fetch: $id $commit"
        [[ -z "$(git -C "$repository" status --porcelain --untracked-files=all)" ]] \
            || die "existing clone has local changes and will not be modified: $repository"

        local head
        head="$(git -C "$repository" rev-parse HEAD)"
        if [[ "$head" != "$commit" ]]; then
            printf '[checkout] %s: %s -> %s\n' "$id" "$head" "$commit" >&2
            git -C "$repository" checkout --detach "$commit"
        else
            printf '[skip clone] %s already has full history at %s\n' "$id" "$commit" >&2
        fi
    else
        printf '[clone] %s -> %s\n' "$url" "$repository" >&2
        local clone_stage
        clone_stage="$(mktemp -d "$repository_parent/.${id##*/}.clone.XXXXXX")"
        local staged_repository="$clone_stage/repository"
        if ! git clone --no-checkout -- "$url" "$staged_repository"; then
            die "clone failed; unpublished staging is preserved for inspection: $clone_stage"
        fi
        if ! git -C "$staged_repository" checkout --detach "$commit"; then
            die "frozen commit checkout failed; unpublished staging is preserved: $clone_stage"
        fi
        if [[ "$(git -C "$staged_repository" rev-parse --is-shallow-repository)" != "false" \
            || "$(git -C "$staged_repository" config --bool --get remote.origin.promisor 2>/dev/null || true)" == "true" ]]; then
            die "new clone is shallow or partial; unpublished staging is preserved: $clone_stage"
        fi
        if [[ -e "$repository" || -L "$repository" ]]; then
            die "repository destination appeared during clone; unpublished staging is preserved: $clone_stage"
        fi
        mv -- "$staged_repository" "$repository"
        rmdir -- "$clone_stage"
    fi

    [[ "$(git -C "$repository" rev-parse --is-shallow-repository)" == "false" ]] \
        || die "snapshot clone is unexpectedly shallow: $repository"
    [[ "$(git -C "$repository" rev-parse HEAD)" == "$commit" ]] \
        || die "snapshot checkout mismatch: $repository"
    [[ -z "$(git -C "$repository" status --porcelain --untracked-files=all)" ]] \
        || die "snapshot clone is not clean: $repository"
    printf '%s\n' "$repository"
}

command -v git >/dev/null 2>&1 || die "git is required"

if [[ "$CLONE_ONLY" != true ]]; then
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
    [[ -f "$CONFIG" ]] || die "config is missing: $CONFIG"
    PYTHON_PREFIX_PATH="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
    NORM4CODE_CLI="$PYTHON_PREFIX_PATH/bin/repo-memory-v3"
    export PATH="$PYTHON_PREFIX_PATH/bin:$PATH"
    "$NORM4CODE_CLI" --help >/dev/null || die "install the core package in the active Python environment"
    if [[ "$MOCK" != true ]]; then
        command -v bwrap >/dev/null 2>&1 && command -v mini >/dev/null 2>&1 \
            || die "real source discovery requires bwrap and mini in the active Python environment"
        "$PYTHON_BIN" -c '
import os, sys
from dotenv import find_dotenv, load_dotenv
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
path = find_dotenv(usecwd=True)
if path:
    load_dotenv(path)
profile = load_execution_model_profile(sys.argv[1])
key = profile.provider.api_key_env
if not key or not os.environ.get(key):
    raise SystemExit("Selected profile credential is unavailable: " + str(key))
' "$MODEL_PROFILE" || die "model-profile credential preflight failed"
    fi
fi

profile_run_key() {
    if [[ "$MOCK" == true ]]; then
        printf 'mock-%s%s\n' "$WORKFLOW_ID" "$CONFIG_SUFFIX"
        return
    fi
    local display
    display="$(basename "$MODEL_PROFILE")"
    display="${display//[^a-zA-Z0-9._-]/_}"
    local digest
    if [[ -f "$MODEL_PROFILE" ]]; then
        digest="$(sha256sum "$MODEL_PROFILE" | cut -c1-12)"
    else
        digest="$(printf '%s' "$MODEL_PROFILE" | sha256sum | cut -c1-12)"
    fi
    printf 'real-%s-%s-%s%s\n' "$display" "$digest" "$WORKFLOW_ID" "$CONFIG_SUFFIX"
}

find_existing_package() {
    local profile_runs="$1"
    local candidate
    shopt -s nullglob
    local manifests=("$profile_runs"/run-*/package/consumers/accepted_only/consumer-manifest.json)
    shopt -u nullglob
    for candidate in "${manifests[@]}"; do
        if [[ -f "${candidate%/consumers/accepted_only/consumer-manifest.json}/canonical/claims.json" ]]; then
            printf '%s\n' "${candidate%/consumers/accepted_only/consumer-manifest.json}"
            return 0
        fi
    done
    return 1
}

new_run_directory() {
    local profile_runs="$1"
    local timestamp
    timestamp="$(date '+%Y-%m-%dT%H%M%S%z')"
    local candidate="$profile_runs/run-$timestamp-$$"
    local suffix=0
    while [[ -e "$candidate" ]]; do
        ((suffix += 1))
        candidate="$profile_runs/run-$timestamp-$$-$suffix"
    done
    mkdir -p "$candidate"
    printf '%s\n' "$candidate"
}

run_logged() {
    local log_file="$1"
    shift
    set +e
    "$@" 2>&1 | tee -a "$log_file"
    local command_status="${PIPESTATUS[0]}"
    set -e
    return "$command_status"
}

extract_snapshot() {
    local id="$1"
    local repository="$2"
    local commit="${SNAPSHOT_COMMIT[$id]}"
    local profile_runs="$RUNS_DIR/$id/$(profile_run_key)"
    mkdir -p "$profile_runs"

    local existing=""
    if existing="$(find_existing_package "$profile_runs")" && [[ "$RERUN" != true ]]; then
        printf '[skip extraction] %s already has a published package: %s\n' "$id" "$existing"
        printf '  use --rerun to create another run without overwriting it\n'
        return 0
    fi

    local run_root
    run_root="$(new_run_directory "$profile_runs")"
    local acquisition="$run_root/acquisition"
    local package="$run_root/package"
    local inference_cache="$run_root/inference-cache"
    local observation_cache="$CACHES_DIR/observation/$id"
    local log_file="$run_root/workflow.log"
    local receipts="$run_root/receipts"
    mkdir -p "$inference_cache" "$observation_cache" "$receipts"
    local run_config="$run_root/config.yaml"
    cp -- "$CONFIG" "$run_config"
    if [[ "$CONFIG_EXPLICIT" == true ]]; then
        [[ "-config-$(sha256sum "$run_config" | cut -d ' ' -f 1)" == "$CONFIG_SUFFIX" ]] \
            || die "config changed during batch; preserved run: $run_root"
    fi

    printf 'target=%s\ncommit=%s\nrepository=%s\nmode=%s\nmodel_profile=%s\nreceipt_schema=%s\n' \
        "$id" "$commit" "$repository" "$([[ "$MOCK" == true ]] && printf mock || printf real)" \
        "$([[ "$MOCK" == true ]] && printf offline-test || printf '%s' "$MODEL_PROFILE")" \
        "norm4code-command-run-receipt-1.0" \
        >"$run_root/RUN_INFO"

    local agent_profile="$REAL_AGENT_PROFILE"
    local -a cli=("$NORM4CODE_CLI")
    local -a discover=("${cli[@]}" discover "$repository" --config "$run_config" --out "$acquisition" --run-receipt "$receipts/discover.json")
    local -a build=("${cli[@]}" build "$repository" --config "$run_config" --acquisition "$acquisition" --out "$package" --observation-cache "$observation_cache" --inference-cache "$inference_cache" --run-receipt "$receipts/build.json")

    if [[ "$MOCK" == true ]]; then
        agent_profile="$MOCK_AGENT_PROFILE"
        discover+=(--agent-profile "$agent_profile" --mock-agent)
        build+=(--agent-profile "$agent_profile" --mock)
    else
        discover+=(--agent-profile "$agent_profile" --model-profile "$MODEL_PROFILE")
        build+=(--agent-profile "$agent_profile")
    fi

    printf '[discover] %s\n' "$id"
    local discover_status=0
    if run_logged "$log_file" "${discover[@]}"; then
        discover_status=0
    else
        discover_status=$?
        printf 'discover_failed=%s\n' "$discover_status" >"$run_root/STATUS"
        printf '[failed] discovery exited %s; preserved non-published run: %s\n' \
            "$discover_status" "$run_root" >&2
        return "$discover_status"
    fi

    printf '[build] %s\n' "$id"
    local build_status=0
    if run_logged "$log_file" "${build[@]}"; then
        build_status=0
    else
        build_status=$?
    fi

    if [[ "$build_status" -le 1 && -f "$package/canonical/claims.json" \
        && -f "$package/consumers/accepted_only/consumer-manifest.json" ]]; then
        if [[ "$build_status" -eq 0 ]]; then
            printf 'complete\n' >"$run_root/STATUS"
            printf '[complete] %s -> %s\n' "$id" "$package"
        else
            printf 'valid_partial\n' >"$run_root/STATUS"
            printf '[valid partial] %s -> %s\n' "$id" "$package"
        fi
        return 0
    fi

    printf 'build_failed=%s\n' "$build_status" >"$run_root/STATUS"
    printf '[failed] build exited %s without a published package; run preserved: %s\n' \
        "$build_status" "$run_root" >&2
    return 2
}

for id in "${TARGETS[@]}"; do
    repository="$(ensure_snapshot_clone "$id")"
    if [[ "$CLONE_ONLY" == true ]]; then
        continue
    fi
    extract_snapshot "$id" "$repository"
done

printf 'Done. Data root: %s\n' "$DATA_ROOT"
