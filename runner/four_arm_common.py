"""Host-only identity, timing and cross-process capacity for coding experiments."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
N4C = ROOT
DATASET = ROOT / "benchmark"
DATA_ROOT = ROOT / "outputs"
PROTOCOL = "cooperbench-four-arm-1.5"
ARMS = ("raw", "codewiki", "accepted_only", "accepted_after_review")
MODEL = "deepseek-v4-flash-260425"
PROFILE = "apiyi-deepseek-v4-flash-260425-nonthinking"
ENDPOINT = "https://vip.apiyi.com/v1"
CONCURRENCY = 20
DOCKER_START_TIMEOUT_OUTCOME = "docker_start_timeout"
INFRASTRUCTURE_MAX_RETRIES = 3


def infrastructure_retry_pending(row: dict, max_retries: int = INFRASTRUCTURE_MAX_RETRIES) -> bool:
    telemetry = row.get("telemetry") or {}
    eligible = row.get("outcome") in {DOCKER_START_TIMEOUT_OUTCOME, "agent_error"}
    if row.get("outcome") == "infrastructure_error":
        eligible = (
            row.get("infrastructure_retry_reason") in {"docker_timeout", "docker_daemon_transient"}
            or (row.get("container_cleanup") or {}).get("status") == "unconfirmed"
            or (row.get("pre_retry_cleanup") or {}).get("status") == "unconfirmed"
        )
    return (
        eligible
        and row.get("infrastructure_retry_round", 0) < max_retries
        and not telemetry.get("violations")
        and telemetry.get("identity_status") != "violation"
    )


def docker_retry_reason(error: BaseException) -> str | None:
    """Classify host Docker failures, not arbitrary agent shell command text."""
    if error.__cause__ is not None:
        return docker_retry_reason(error.__cause__)
    args = getattr(error, "cmd", None)
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        return None
    if Path(str(args[0])).name != "docker" or args[1] not in {"run", "inspect", "exec", "cp", "ps", "rm", "image"}:
        return None
    if isinstance(error, subprocess.TimeoutExpired):
        return "docker_timeout"
    if isinstance(error, subprocess.CalledProcessError):
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if any(message in stderr.lower() for message in (
            "cannot connect to the docker daemon", "error during connect",
            "context deadline exceeded", "i/o timeout", "tls handshake timeout",
            "resource temporarily unavailable", "timed out waiting", "timeout waiting",
        )):
            return "docker_daemon_transient"
    return None


def dataset_path(config: dict | None = None) -> Path:
    return DATASET


def runtime_for(config: dict | None = None):
    import environment
    return environment


def feature_path(pool: dict, feature: int) -> Path:
    from dataset import task_for_pool
    return DATASET / task_for_pool(pool, feature)["path"]


def credential_env(config: dict) -> str:
    return config.get("credential_env", "APIYI_API_KEY")


def response_model(config: dict) -> str:
    return config.get("response_model", config["model"])


def model_kwargs(config: dict) -> dict:
    """Frozen SDK request arguments; credentials are attached only in the worker."""
    extra_body = dict(config["extra_body"] if "extra_body" in config else {"thinking": config["thinking"]})
    kwargs = {"api_base": config["endpoint"], "extra_body": extra_body, "drop_params": False}
    if config.get("temperature_transport") == "extra_body":
        # LiteLLM applies OpenAI's GPT-5 compatibility policy to any model with
        # this name before contacting an OpenAI-compatible third-party endpoint.
        # Preserve the frozen temperature as an actual JSON request-body field
        # while avoiding that provider-specific client-side transformation.
        extra_body["temperature"] = config["temperature"]
    else:
        kwargs["temperature"] = config["temperature"]
    if "max_tokens" in config:
        kwargs["max_tokens"] = config["max_tokens"]
    if "request_timeout_seconds" in config:
        kwargs.update(timeout=config["request_timeout_seconds"], num_retries=0)
    if "api_timeout_seconds" in config:
        kwargs.update(timeout=float(config["api_timeout_seconds"]),
                      max_retries=config["api_sdk_max_retries"], num_retries=0)
    return kwargs


def read_credential(config: dict) -> str | None:
    from dotenv import dotenv_values
    return os.environ.get(credential_env(config)) or dotenv_values(ROOT / ".env").get(credential_env(config))


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command(args: list[str], *, timeout: int = 180, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr[-1500:]}")
    return result.stdout


def pool_key(pool: dict) -> str:
    return f"{pool['repository']}/task{pool['task_id']}"


def task_key(pool: dict, feature: int) -> str:
    return pool_key(pool).replace("/", "-") + f"-feature{feature}"


@contextlib.contextmanager
def exclusive(path: Path, *, blocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def global_slot(slot_root: Path | None = None, *, capacity: int = CONCURRENCY) -> Iterator[int]:
    """A configurable prefix of the twenty shared kernel locks."""
    if type(capacity) is not int or not 1 <= capacity <= CONCURRENCY:
        raise ValueError(f"Global concurrency must be between 1 and {CONCURRENCY}")
    root = slot_root or DATA_ROOT / ".coding-task-slots"
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    selected = None
    try:
        for index in range(capacity):
            handle = (root / f"slot-{index:02}.lock").open("a+")
            handles.append(handle)
        # There is no portable "wait for any flock" primitive. Re-scan all
        # slots with a short sleep instead of blocking forever on one selected
        # lock: otherwise waiters can queue behind one busy slot while the rest
        # of the shared pool is already idle.
        start = 0
        while selected is None:
            for offset in range(capacity):
                index = (start + offset) % capacity
                try:
                    fcntl.flock(handles[index], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = index
                    break
                except BlockingIOError:
                    pass
            if selected is None:
                time.sleep(0.1)
                start = (start + 1) % capacity
        yield selected
    finally:
        for handle in handles:
            handle.close()


class Timings:
    def __init__(self) -> None:
        self.phases: dict[str, dict] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started, monotonic = utc(), time.monotonic()
        status = "complete"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self.phases[name] = {
                "started_at": started, "finished_at": utc(),
                "elapsed_seconds": time.monotonic() - monotonic, "status": status,
            }


# Same bytes run in the host clone and the actual initial task container.
# Hash worktree bytes (not Git's index/stat cache), including nonignored untracked
# files. Ignored image dependencies are outside the repository knowledge snapshot.
SNAPSHOT_SCRIPT = r'''
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path.cwd()
def git(*args):
    return subprocess.check_output(["git", *args])
paths = sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")) - {b""})
files = []
for raw in paths:
    name = os.fsdecode(raw)
    p = root / name
    if p.is_symlink():
        files.append([name, "symlink", os.readlink(p)])
    elif p.is_file():
        files.append([name, "executable" if p.stat().st_mode & 0o111 else "file", hashlib.sha256(p.read_bytes()).hexdigest()])
    else:
        raise RuntimeError("Nonregular or missing tracked file: " + name)
print(json.dumps({"head": git("rev-parse", "HEAD").decode().strip(), "files": files}, sort_keys=True))
'''


def repository_identity(repository: Path) -> dict:
    return json.loads(command([__import__("sys").executable, "-c", SNAPSHOT_SCRIPT], cwd=repository))
