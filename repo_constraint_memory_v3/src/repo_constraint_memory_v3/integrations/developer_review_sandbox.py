"""Linux Codex tool isolation; the authenticated Codex host stays outside it.

Named permissions are essential: legacy ``--sandbox read-only`` allows ambient
host reads. A no-model preflight verifies the installed CLI enforces the policy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any

PROFILE_NAME = "norm4code_review"


def _native_binary(binary: str) -> Path:
    resolved = Path(shutil.which(binary) or binary).resolve(strict=True)
    with resolved.open("rb") as stream:
        if stream.read(4) == b"\x7fELF":
            return resolved
    package = resolved.parent.parent
    matches = list(package.glob("node_modules/@openai/codex-linux-*/vendor/*/bin/codex"))
    if len(matches) != 1:
        raise ValueError("Cannot resolve one native Linux Codex binary for sandbox re-execution")
    return matches[0].resolve(strict=True)


def sandbox_config(binary: str, snapshot: Path, inputs: Path, work: Path) -> dict[str, Any]:
    """Return CLI overrides; do not combine these with legacy --sandbox flags."""
    if sys.platform != "linux":
        raise ValueError("Isolated developer review currently requires Linux")
    roots = [path.resolve(strict=True) for path in (snapshot, inputs, work)]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("Snapshot, input, and writable work roots must be disjoint")
    snapshot, inputs, work = roots
    native = _native_binary(binary)
    bundled_rg = native.parent.parent / "codex-path" / "rg"
    filesystem = {":root": "deny", ":minimal": "read", str(native): "read",
                  str(snapshot): "read", str(inputs): "read", str(work): "write"}
    tool_path = "/usr/bin:/bin"
    if bundled_rg.is_file():
        filesystem[str(bundled_rg)] = "read"
        tool_path += ":" + str(bundled_rg.parent)
    for name in (".codex", ".agents"):
        protected = work / name
        if protected.is_symlink():
            raise ValueError("Review configuration directories must not be symlinks")
        protected.mkdir(exist_ok=True)
        filesystem[str(protected)] = "read"
    runtime = work / ".codex" / "runtime"
    runtime.mkdir(exist_ok=True)
    helper = runtime / "codex-linux-sandbox"
    if helper.is_symlink():
        if helper.resolve() != native:
            raise ValueError("Unexpected sandbox runtime helper target")
    elif helper.exists():
        raise ValueError("Sandbox runtime helper must be a native binary symlink")
    else:
        helper.symlink_to(native)
    tool_path += ":" + str(runtime)
    (work / "tmp").mkdir(exist_ok=True)
    return {
        "default_permissions": PROFILE_NAME,
        f"permissions.{PROFILE_NAME}.filesystem": filesystem,
        f"permissions.{PROFILE_NAME}.network.enabled": False,
        "approval_policy": "never",
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": {"PATH": tool_path, "HOME": str(work),
                                         "TMPDIR": str(work / "tmp"), "LANG": "C.UTF-8"},
    }


def config_arguments(config: dict[str, Any]) -> list[str]:
    """Encode structured values as TOML, without a shell or shell interpolation."""
    def encode(value: Any) -> str:
        if isinstance(value, dict):
            return "{" + ",".join(json.dumps(key) + "=" + encode(item) for key, item in value.items()) + "}"
        return json.dumps(value, ensure_ascii=False)
    return [part for key, value in config.items() for part in ("-c", key + "=" + encode(value))]


def verify_sandbox(binary: str, environment: dict[str, str], snapshot: Path, inputs: Path, work: Path) -> dict[str, Any]:
    """Probe real OS boundaries with synthetic data; never reads real credentials.

    The finite timeout protects this local infrastructure check only. It is not
    a reviewer execution budget. Any missing capability or failed probe aborts.
    """
    config = sandbox_config(binary, snapshot, inputs, work)
    with tempfile.TemporaryDirectory(prefix="norm4code-review-isolation-") as temporary:
        outside = Path(temporary) / "host-secret"
        outside.write_text("synthetic isolation sentinel", encoding="utf-8")
        link = work / ".isolation-symlink"
        link.symlink_to(outside)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        payload = {"outside": str(outside), "link": str(link), "work": str(work.resolve()),
                   "snapshot": str(snapshot.resolve()), "inputs": str(inputs.resolve()),
                   "parent_pid": os.getpid(), "port": listener.getsockname()[1]}
        probe = r'''import json, os, pathlib, socket, sys
p = json.loads(sys.argv[1])
checks = {}
for name, path in [("host_read_denied", p["outside"]), ("symlink_escape_denied", p["link"]),
                   ("proc_parent_hidden", "/proc/" + str(p["parent_pid"]) + "/environ"),
                   ("proc_root_escape_denied", "/proc/self/root" + p["outside"])]:
    try:
        pathlib.Path(path).read_bytes()
        checks[name] = False
    except OSError:
        checks[name] = True
for name in ("snapshot", "inputs"):
    checks[name + "_readable"] = pathlib.Path(p[name]).is_dir()
for name, root in [("snapshot", p["snapshot"]), ("inputs", p["inputs"]),
                   ("configuration", p["work"] + "/.codex"), ("skills", p["work"] + "/.agents")]:
    target = pathlib.Path(root) / ".isolation-write-probe"
    try:
        target.write_text("probe")
        target.unlink()
        checks[name + "_readonly"] = False
    except OSError:
        checks[name + "_readonly"] = True
out = pathlib.Path(p["work"]) / ".isolation-work-probe"
out.write_text("probe")
checks["work_writable"] = out.read_text() == "probe"
out.unlink()
checks["environment_clean"] = "NORM4CODE_ISOLATION_SENTINEL" not in os.environ
s = None
try:
    s = socket.socket()
    s.settimeout(1)
    s.connect(("127.0.0.1", p["port"]))
    checks["network_denied"] = False
except OSError:
    checks["network_denied"] = True
finally:
    if s is not None:
        s.close()
print(json.dumps(checks))
sys.exit(0 if all(checks.values()) else 1)
'''
        command = [str(_native_binary(binary)), "sandbox", "-P", PROFILE_NAME, "-C", str(work.resolve())]
        command += config_arguments(config)
        command += ["/usr/bin/python3", "-c", probe, json.dumps(payload)]
        try:
            # `codex sandbox` inherits its environment directly. The exec adapter
            # uses the same explicit environment for model-created shells.
            probe_environment = {"PATH": "/usr/bin:/bin", "HOME": environment.get("HOME", str(Path.home())),
                                 "NORM4CODE_ISOLATION_SENTINEL": "synthetic environment secret"}
            # Host authentication/config lookup remains outside the tool sandbox.
            for key in ("CODEX_HOME",):
                if key in environment:
                    probe_environment[key] = environment[key]
            result = subprocess.run(command, env=probe_environment, capture_output=True,
                                    text=True, timeout=30, check=False)
            try:
                checks = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                checks = {}
            if result.returncode or not checks or not all(value is True for value in checks.values()):
                raise ValueError("Codex sandbox isolation preflight failed: " + result.stderr[-2000:])
            return {"mechanism": "codex-linux-permission-profile", "profile": PROFILE_NAME,
                    "checks": checks, "real_model_called": False}
        finally:
            listener.close()
            link.unlink(missing_ok=True)
