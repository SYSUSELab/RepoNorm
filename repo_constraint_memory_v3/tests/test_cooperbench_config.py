from pathlib import Path
import subprocess

from repo_constraint_memory_v3.builder import load_build_config
from repo_constraint_memory_v3.discovery import build_workspace_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_variant_changes_only_single_file_cap(tmp_path):
    original = load_build_config(ROOT / "repo_constraint_memory_v3/configs/hardened.yaml")
    variant = load_build_config(ROOT / "repo_constraint_memory_v3/configs/hardened-manifest-5mb.yaml")
    expected = original.model_dump()
    expected["discovery"]["max_manifest_file_bytes"] = 5_000_000
    assert variant.model_dump() == expected
    (tmp_path / "example.ipynb").write_text(" " * 4_172_850)
    old = build_workspace_manifest(tmp_path, original.discovery, candidate_paths=["example.ipynb"])
    new = build_workspace_manifest(tmp_path, variant.discovery, candidate_paths=["example.ipynb"])
    assert "max_manifest_file_bytes_exceeded" in old.incomplete_reasons
    assert not new.incomplete_reasons


def test_config_namespace_separates_published_package_lookup(tmp_path):
    script = (ROOT / "run_cooperbench_norm4code.sh").read_text()
    functions = script[script.index("profile_run_key() {"):script.index("new_run_directory() {")]
    # Exercise the production key and published-package lookup together.
    command = functions + """
MOCK=true
WORKFLOW_ID=test
CONFIG_SUFFIX=-config-old
old_key="$(profile_run_key)"
mkdir -p "$1/$old_key/run-one/package/consumers/accepted_only" "$1/$old_key/run-one/package/canonical"
touch "$1/$old_key/run-one/package/consumers/accepted_only/consumer-manifest.json" "$1/$old_key/run-one/package/canonical/claims.json"
find_existing_package "$1/$old_key" >/dev/null || exit 10
CONFIG_SUFFIX=-config-new
new_key="$(profile_run_key)"
if find_existing_package "$1/$new_key"; then exit 11; fi
MOCK=false
MODEL_PROFILE=deepseek-flash-nonthinking
first="$(profile_run_key)"
CONFIG_SUFFIX=-config-old
test "$first" != "$(profile_run_key)"
"""
    subprocess.run(["bash", "-c", command, "test", str(tmp_path)], check=True)
