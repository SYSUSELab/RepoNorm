"""Prepare reusable, authenticated Engineering-77 Docker evaluation resources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'evaluation/checks/extended'
DEFAULT = ROOT / 'outputs/evaluation-resources/extended'
STANDARD_LOCKS = ROOT / 'evaluation/checks/standard/tool-locks'
LOCKS = {
    'outlines-2023': FROZEN / 'tool-locks/OUT-7149a5e613b9.txt',
    'click-2024': STANDARD_LOCKS / 'CLI.txt',
    'click-2025': FROZEN / 'tool_locks/click-2025.txt',
    'jinja-2021': STANDARD_LOCKS / 'JIN.txt',
}
# Historical full environment locks were not retained for these profiles. These
# explicit reconstruction pins must never be represented as historical receipts.
# DSP Ruff is the version installed in the frozen DSP runtime image; the old
# host-local static-tool version is unavailable, so this is a new resource identity.
RECONSTRUCTED = {
    'outlines-2025': 'ruff==0.9.1\nmypy==1.14.1\ntypes-tqdm\ntypes-Pillow\n',
    'dspy-2025': 'ruff==0.16.2\n',
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def docker(*args: str) -> str:
    return subprocess.check_output(['docker', *args], text=True).strip()


def pins(value: str) -> dict:
    return {re.sub(r'[-_.]+', '-', name).lower(): version
            for line in value.splitlines() if '==' in line and not line.startswith('#')
            for name, version in [line.split('==', 1)]}


def requirements(profile: str) -> str:
    result = LOCKS[profile].read_text() if profile in LOCKS else RECONSTRUCTED[profile]
    if profile == 'outlines-2023':
        result += '\ntypes-tqdm\ntypes-Pillow\n'
    if profile == 'jinja-2021':
        result += '\npyupgrade==3.3.1\ntokenize_rt==6.2.0\n'
    return result


def verify_resources(resources: dict, tasks: list[dict]) -> None:
    """Authenticate selected runtime images, bundles, and static tool receipts."""
    seen = set()
    for task in tasks:
        snapshot, profile = task['snapshot'], task['tool_profile']
        image = resources['runtime_images'][task['id']]
        if image != task['image_id']:
            raise ValueError(f'Runtime image mismatch: {task["id"]}')
        if (snapshot, profile, task['image_id']) in seen:
            continue
        seen.add((snapshot, profile, task['image_id']))
        if docker('image', 'inspect', image, '--format', '{{.Id}}') != image:
            raise ValueError(f'Runtime image mismatch: {snapshot}')
        bundle = Path(resources['snapshots'][snapshot])
        if digest(bundle) != resources['snapshot_sha256'][snapshot]:
            raise ValueError(f'Snapshot bundle mismatch: {snapshot}')
        heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True).strip()
        if heads != snapshot + ' HEAD':
            raise ValueError(f'Snapshot HEAD mismatch: {snapshot}')
        static = resources['static_images'][profile]
        receipt = resources['tool_receipts'][profile]
        if docker('image', 'inspect', static, '--format', '{{.Id}}') != static or receipt['image_id'] != static:
            raise ValueError(f'Static image mismatch: {profile}')
        expected = hashlib.sha256(requirements(profile).encode()).hexdigest()
        if receipt['requirements_sha256'] != expected:
            raise ValueError(f'Tool requirements mismatch: {profile}')
        installed = Path(receipt['installed_path'])
        if digest(installed) != receipt['installed_sha256']:
            raise ValueError(f'Installed tool receipt mismatch: {profile}')
        actual = pins(installed.read_text())
        if any(actual.get(k) != v for k, v in pins(requirements(profile)).items()):
            raise ValueError(f'Installed tool pins mismatch: {profile}')


def write_prepared(destination: Path, record: dict) -> None:
    path = destination / 'prepared.json'
    temp = destination / 'prepared.json.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(path)


def prepare_snapshot(task: dict, destination: Path, registry: str) -> tuple[str, str]:
    image, snapshot = task['image_id'], task['snapshot']
    result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True)
    if result.returncode:
        ref = (registry.rstrip('/') + '/' if registry else '') + task['image_ref']
        with (destination / (snapshot + '.pull.log')).open('w') as log:
            subprocess.run(['docker', 'pull', ref], stdout=log, stderr=subprocess.STDOUT, check=True)
    if docker('image', 'inspect', image, '--format', '{{.Id}}') != image:
        raise ValueError(f'Pulled runtime identity mismatch: {snapshot}')
    head = docker('run', '--rm', '--network', 'none', '--entrypoint', '', image,
                  'git', '-C', '/workspace/repo', 'rev-parse', 'HEAD')
    if head != snapshot:
        raise ValueError(f'Runtime snapshot mismatch: {snapshot}')
    folder = destination / 'snapshots'
    folder.mkdir(exist_ok=True)
    bundle = folder / (snapshot + '.bundle')
    if not bundle.exists():
        docker('run', '--rm', '--network', 'none', '--entrypoint', '',
               '-v', f'{folder.resolve()}:/export', image, 'git', '-C', '/workspace/repo',
               'bundle', 'create', '/export/' + snapshot + '.bundle', 'HEAD')
    heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True).strip()
    if heads != snapshot + ' HEAD':
        raise ValueError(f'Existing bundle identity mismatch: {snapshot}')
    return image, str(bundle.resolve())


def build_tools(profile: str, destination: Path, base: str) -> tuple[str, dict]:
    folder = destination / 'tool-build' / profile
    folder.mkdir(parents=True, exist_ok=True)
    lock = requirements(profile)
    (folder / 'requirements.txt').write_text(lock)
    # No target source/configuration is used at image build time.
    recipe = (f'FROM {base}\n'
              'RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm && rm -rf /var/lib/apt/lists/*\n'
              'RUN python -m venv /opt/tools\n'
              'COPY requirements.txt /tmp/requirements.txt\n'
              'RUN /opt/tools/bin/pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -r /tmp/requirements.txt\n'
              'ENV PATH=/opt/tools/bin:$PATH\n'
              'ENV XDG_CACHE_HOME=/opt/static/cache\n')
    if profile.startswith('click-'):
        recipe += 'RUN /opt/tools/bin/pyright --version\n'
    (folder / 'Dockerfile').write_text(recipe)
    tag = 'n4c-norm77-tools:' + profile
    with (folder / 'build.log').open('w') as log:
        subprocess.run(['docker', 'build', '--network', 'host', '-t', tag, str(folder)],
                       env=dict(os.environ, DOCKER_BUILDKIT='0'), stdout=log, stderr=subprocess.STDOUT, check=True)
    image = docker('image', 'inspect', tag, '--format', '{{.Id}}')
    actual = docker('run', '--rm', '--network', 'none', '--entrypoint', '', image,
                    '/opt/tools/bin/python', '-m', 'pip', 'freeze', '--all') + '\n'
    installed = destination / (profile + '-installed.txt')
    installed.write_text(actual)
    if any(pins(actual).get(k) != v for k, v in pins(lock).items()):
        raise ValueError(f'Installed versions differ: {profile}')
    if profile.startswith('click-'):
        docker('run', '--rm', '--network', 'none', '--entrypoint', '', image, '/opt/tools/bin/pyright', '--version')
    receipt = {'image_id': image, 'requirements_sha256': hashlib.sha256(lock.encode()).hexdigest(),
               'installed_sha256': digest(installed), 'installed_path': str(installed.resolve()),
               'historical_lock_available': profile in LOCKS,
               'historical_full_lock_available': profile in LOCKS and profile not in {'click-2025', 'outlines-2023'},
               'reconstruction': profile in RECONSTRUCTED,
               'base_image_id': docker('image', 'inspect', base, '--format', '{{.Id}}'),
               'python': docker('run', '--rm', '--network', 'none', '--entrypoint', '', image, '/opt/tools/bin/python', '--version'),
               'limitations': ([f'Historical full environment lock unavailable; {profile} uses explicitly reconstructed pins.']
                               if profile in RECONSTRUCTED else
                               ['Historical direct tool pins are retained; transitive dependencies are captured by this new installed receipt.']
                               if profile == 'click-2025' else
                               ['Historical lock omitted the snapshot pre-commit hook additional dependencies types-tqdm/types-Pillow; this environment supplies and records them.']
                               if profile == 'outlines-2023' else [])}
    return image, receipt

