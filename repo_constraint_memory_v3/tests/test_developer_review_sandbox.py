"""Offline isolation checks never invoke a model or read real credentials."""
import os
from pathlib import Path
import shutil
import sys

import pytest

from repo_constraint_memory_v3.integrations.developer_review_sandbox import (
    PROFILE_NAME, config_arguments, sandbox_config, verify_sandbox,
)


def _roots(tmp_path):
    roots = [tmp_path / name for name in ("repository", "input", "work")]
    for root in roots:
        root.mkdir()
    return roots


def _fake_binary(tmp_path):
    binary = tmp_path / "codex"
    binary.write_bytes(b"\x7fELF")
    return str(binary)


def test_restricted_profile_has_no_ambient_reads_or_environment(tmp_path):
    repository, inputs, work = _roots(tmp_path)
    config = sandbox_config(_fake_binary(tmp_path), repository, inputs, work)
    permissions = config[f"permissions.{PROFILE_NAME}.filesystem"]
    assert permissions[":root"] == "deny"
    assert permissions[str(repository)] == permissions[str(inputs)] == "read"
    assert permissions[str(work)] == "write"
    assert permissions[str(work / ".codex")] == "read"
    assert config[f"permissions.{PROFILE_NAME}.network.enabled"] is False
    assert config["shell_environment_policy.inherit"] == "none"
    assert config["approval_policy"] == "never"
    assert "sandbox_mode" not in config
    assert (work / ".codex/runtime/codex-linux-sandbox").resolve() == Path(tmp_path / "codex")


def test_overlapping_roots_rejected(tmp_path):
    repository, inputs, work = _roots(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        sandbox_config(_fake_binary(tmp_path), tmp_path, inputs, work)


def test_configuration_symlink_rejected(tmp_path):
    repository, inputs, work = _roots(tmp_path)
    (work / ".codex").symlink_to(inputs, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        sandbox_config(_fake_binary(tmp_path), repository, inputs, work)


def test_toml_encoding_preserves_unusual_paths():
    args = config_arguments({"test": {"/tmp/a\"b\\c": "read"}, "enabled": False})
    assert args == ["-c", 'test={"/tmp/a\\"b\\\\c"="read"}', "-c", "enabled=false"]


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("codex") is None,
                    reason="Requires installed Linux Codex CLI; no model is called")
def test_installed_codex_enforces_boundaries_without_model(tmp_path):
    roots = _roots(tmp_path)
    receipt = verify_sandbox(shutil.which("codex"), dict(os.environ), *roots)
    assert receipt["real_model_called"] is False
    assert receipt["checks"]
    assert all(receipt["checks"].values())
    assert not (roots[2] / ".isolation-symlink").exists()
