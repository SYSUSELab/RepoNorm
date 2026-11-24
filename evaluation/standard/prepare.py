"""Prepare exact-identity assets; downloads occur here, never during scoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import html
import re
import urllib.parse
import urllib.error
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "evaluation/checks/standard"
DEFAULT = ROOT / "outputs/evaluation-resources/standard"
BASE = "sha256:db825c749364f7f06c8fb165b360d9de94ba5c14727e2e446c1b4202a30224e7"
ENCODINGS = ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"] + [
    "encodings/" + x + ".tiktoken" for x in ("r50k_base", "p50k_base", "cl100k_base", "o200k_base")]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fetch(url: str, destination: Path, expected: str, size: int | None = None) -> None:
    if destination.exists():
        if digest(destination) != expected or (size is not None and destination.stat().st_size != size):
            raise ValueError(f"Existing asset mismatch: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if digest(temporary) != expected or (size is not None and temporary.stat().st_size != size):
                raise ValueError(f"Downloaded asset mismatch: {destination.name}")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise


def wheel(name: str, expected: str, directory: Path) -> None:
    destination = directory / name
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Wheel mismatch: {name}")
        return
    package = name.split("-")[0].replace("_", "-")
    with urllib.request.urlopen(f"https://mirrors.aliyun.com/pypi/simple/{package}/", timeout=60) as response:
        page = response.read().decode()
    links = [html.unescape(x) for x in re.findall(r'href=["\']([^"\']+)', page)]
    url = next((x for x in links if urllib.parse.unquote(urllib.parse.urlsplit(x).path).endswith("/" + name)), None)
    if url is None:
        raise ValueError(f"Frozen wheel unavailable from mirror: {name}")
    fetch(urllib.parse.urljoin(f"https://mirrors.aliyun.com/pypi/simple/{package}/", url.split("#")[0]), destination, expected)


def prepare_assets(destination: Path) -> dict:
    paths = {"tiktoken_cache": destination / "TIK-cache", "tiktoken_wheels": destination / "TIK-wheels",
             "outlines_wheels": destination / "OUT-wheels"}
    manifest = json.loads((FROZEN / "runtime_assets/tiktoken-cache.json").read_text())
    for suffix in ENCODINGS:
        url = "https://openaipublic.blob.core.windows.net/" + suffix
        filename = hashlib.sha1(url.encode()).hexdigest()
        item = manifest["files"][filename]
        cached = ROOT / "outputs/runtime/TIK-cache" / filename
        target = paths["tiktoken_cache"] / filename
        if (not target.exists() and cached.is_file() and digest(cached) == item["sha256"]
                and cached.stat().st_size == item["bytes"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        fetch(url, target, item["sha256"], item["bytes"])
    jobs = []
    for key, filename in [("tiktoken_wheels", "tiktoken-numpy-wheel.json"), ("outlines_wheels", "outlines-cpu-wheels.json")]:
        entries = json.loads((FROZEN / "runtime_assets" / filename).read_text())["files"]
        paths[key].mkdir(parents=True, exist_ok=True)
        for name, spec in entries.items():
            expected = spec["sha256"] if isinstance(spec, dict) else spec
            cached = ROOT / "outputs/runtime/TIK-wheels" / name
            if cached.exists() and digest(cached) == expected and not (paths[key] / name).exists():
                shutil.copyfile(cached, paths[key] / name)
            jobs.append((name, expected, paths[key]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: wheel(*args), jobs))
    return {key: str(value.resolve()) for key, value in paths.items()}


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True).strip()


def snapshots(destination: Path) -> tuple[dict, dict]:
    tasks = json.loads((FROZEN / "norms.json").read_text())["tasks"]
    images, bundles = {}, {}
    folder = destination / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        code = task["code"]
        if code in images:
            continue
        image = docker("image", "inspect", task["image_id"], "--format", "{{.Id}}")
        if image != task["image_id"]:
            raise ValueError("Runtime image mismatch")
        head = docker("run", "--rm", "--network", "none", "--entrypoint", "", image,
                      "git", "-C", "/workspace/repo", "rev-parse", "HEAD")
        if head != task["snapshot"]:
            raise ValueError("Snapshot mismatch")
        bundle = folder / f"{code}.bundle"
        if not bundle.exists():
            docker("run", "--rm", "--network", "none", "--entrypoint", "", "-v", f"{folder.resolve()}:/export", image,
                   "git", "-C", "/workspace/repo", "bundle", "create", f"/export/{code}.bundle", "HEAD")
        heads = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)], text=True)
        if heads.strip() != task["snapshot"] + " HEAD":
            raise ValueError("Bundle identity mismatch")
        images[code], bundles[code] = image, str(bundle.resolve())
    return images, bundles


def verify_resources(resources: dict, codes: list[str] | None = None, *, checks: Path = FROZEN) -> None:
    """Fail closed on supplied identities; selected-code checks allow staged smoke runs."""
    selected = set(codes or resources["runtime_images"])
    tasks = json.loads((checks / "norms.json").read_text())["tasks"]
    for code in selected:
        task = next(x for x in tasks if x["code"] == code)
        image = resources["runtime_images"][code]
        if image != task["image_id"] or docker("image", "inspect", image, "--format", "{{.Id}}") != image:
            raise ValueError(f"Runtime image mismatch: {code}")
        bundle = Path(resources["snapshots"][code])
        if digest(bundle) != resources["snapshot_sha256"][code]:
            raise ValueError(f"Snapshot bundle mismatch: {code}")
        heads = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)], text=True)
        if heads.strip() != task["snapshot"] + " HEAD":
            raise ValueError(f"Snapshot HEAD mismatch: {code}")
        if code != "TIK":
            static = resources["static_images"][code]
            if docker("image", "inspect", static, "--format", "{{.Id}}") != static:
                raise ValueError(f"Static image mismatch: {code}")
            if "tool_receipts" in resources:
                receipt = resources["tool_receipts"][code]
                folder = bundle.parent.parent
                if (receipt["image_id"] != static
                        or receipt["requirements_sha256"] != digest(folder / (code + "-tools.txt"))
                        or receipt["installed_sha256"] != digest(folder / (code + "-installed.txt"))):
                    raise ValueError(f"Static tool receipt mismatch: {code}")
    assets = []
    if "TIK" in selected:
        assets += [("tiktoken_cache", "tiktoken-cache.json"), ("tiktoken_wheels", "tiktoken-numpy-wheel.json")]
    if "OUT" in selected:
        assets += [("outlines_wheels", "outlines-cpu-wheels.json")]
    for key, name in assets:
        manifest = json.loads((checks / "runtime_assets" / name).read_text())
        for filename, spec in manifest["files"].items():
            path = Path(resources[key]) / filename
            expected = spec["sha256"] if isinstance(spec, dict) else spec
            if digest(path) != expected or (isinstance(spec, dict) and path.stat().st_size != spec["bytes"]):
                raise ValueError(f"Runtime asset mismatch: {filename}")


def write_prepared(destination: Path, update: dict) -> None:
    import fcntl
    with (destination / ".prepare.lock").open("a") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        path = destination / "prepared.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        for key, value in update.items():
            if key in ("static_images", "tool_receipts"):
                record[key] = record.get(key, {}) | value
            else:
                record[key] = value
        temp = destination / "prepared.json.tmp"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(path)


def build_tools(destination: Path, tool_base_image: str = BASE) -> dict:
    """Reconstruct tools in Python 3.10; retain the missing historical OUT-lock limitation."""
    import os
    # Resolve an explicitly chosen image once; subsequent Dockerfiles use its ID.
    if subprocess.run(["docker", "image", "inspect", tool_base_image], capture_output=True).returncode:
        subprocess.run(["docker", "pull", tool_base_image], check=True)
    base_identity = docker("image", "inspect", tool_base_image, "--format", "{{.Id}}")
    python_version = docker("run", "--rm", "--network", "none", "--entrypoint", "",
                            base_identity, "python", "-c",
                            "import platform; print(platform.python_version())")
    if not python_version.startswith("3.10."):
        raise ValueError("Static tool base requires Python 3.10")
    existing_path = destination / "prepared.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    reusable = {}
    for code, image in existing.get("static_images", {}).items():
        receipt = existing.get("tool_receipts", {}).get(code, {})
        lock = destination / (code + "-tools.txt")
        installed = destination / (code + "-installed.txt")
        source = FROZEN / "tool-locks" / (code + ".txt")
        if (receipt.get("image_id") != image or not lock.exists()
                or not installed.exists()
                or receipt.get("installed_sha256") != digest(installed)
                or receipt.get("base_image_id", BASE) != base_identity
                or receipt.get("requirements_sha256") != digest(lock)
                or (source.exists() and source.read_bytes() != lock.read_bytes())
                or (code == "CLI" and not receipt.get("pyright_cache_seeded"))):
            continue
        check = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                               text=True, capture_output=True)
        if check.returncode == 0 and check.stdout.strip() == image:
            reusable[code] = image
    if set(reusable) == {"CLI", "DEQ", "JIN", "OUT"}:
        return reusable
    folder = destination / "tool-build"
    folder.mkdir(parents=True, exist_ok=True)
    def build(tag: str) -> str:
        with (folder / (tag.split(":")[-1] + ".log")).open("w") as log:
            subprocess.run(["docker", "build", "--network", "host", "-t", tag, str(folder)],
                           env=dict(os.environ, DOCKER_BUILDKIT="0"), stdout=log, stderr=subprocess.STDOUT, check=True)
        return docker("image", "inspect", tag, "--format", "{{.Id}}")
    (folder / "Dockerfile").write_text(f"FROM {base_identity}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm && rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m venv /opt/tools\n")
    base = build("n4c-norm-tools:py310-base")
    images = dict(reusable)
    receipts = {code: existing["tool_receipts"][code] for code in reusable}
    for code in ("CLI", "DEQ", "JIN", "OUT"):
        if code in reusable:
            continue
        source = FROZEN / "tool-locks" / (code + ".txt")
        lock = destination / (code + "-tools.txt")
        if source.exists():
            lock.write_bytes(source.read_bytes())
        elif not lock.exists():
            lock.write_text("ruff==0.9.1\nmypy==1.14.1\ntypes-tqdm\ntypes-Pillow\n")
        shutil.copyfile(lock, folder / "requirements.txt")
        no_deps = "--no-deps" if source.exists() else ""
        (folder / "Dockerfile").write_text(f"FROM {base}\nCOPY requirements.txt /tmp/requirements.txt\n"
            f"RUN /opt/tools/bin/pip install --index-url https://mirrors.aliyun.com/pypi/simple/ {no_deps} -r /tmp/requirements.txt\n"
            "ENV PATH=/opt/tools/bin:$PATH\n" +
            ("RUN XDG_CACHE_HOME=/opt/static/cache /opt/tools/bin/pyright --version\n" if code == "CLI" else ""))
        image = build("n4c-norm-tools:" + code.lower() + "-formal003")
        if code == "CLI":
            version = docker("run", "--rm", "--network", "none", "--entrypoint", "",
                             "-e", "XDG_CACHE_HOME=/opt/static/cache", image,
                             "/opt/tools/bin/pyright", "--version")
            if version != "pyright 1.1.386":
                raise ValueError("Offline Pyright bootstrap identity mismatch")
        actual = docker("run", "--rm", "--network", "none", "--entrypoint", "", image,
                        "/opt/tools/bin/pip", "freeze", "--all") + "\n"
        (destination / (code + "-installed.txt")).write_text(actual)
        if source.exists():
            def pins(text: str) -> dict:
                return {re.sub(r"[-_.]+", "-", name).lower(): version
                        for line in text.splitlines() if "==" in line and not line.startswith("#")
                        for name, version in [line.split("==", 1)]}
            if pins(source.read_text()) != pins(actual):
                raise ValueError(f"Installed tools differ from frozen lock: {code}")
        if code == "OUT":
            lock.write_text(actual)
        images[code] = image
        receipts[code] = {"image_id": image, "requirements_sha256": digest(lock),
                          "installed_sha256": digest(destination / (code + "-installed.txt")),
                          "historical_lock_available": source.exists(), "python": python_version,
                          "base_image_id": base_identity, "base_image_requested": tool_base_image,
                          "pyright_cache_seeded": code == "CLI"}
        write_prepared(destination, {"static_images": images, "tool_receipts": receipts})
    (destination / "tool-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return images
