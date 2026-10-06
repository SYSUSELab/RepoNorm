"""Read before/after patch blobs without running repository code."""
import os
import subprocess
from pathlib import Path
def git(repo: Path, index: Path, *args: str, input: bytes | None = None) -> bytes:
    env = {**os.environ, 'GIT_INDEX_FILE': str(index), 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null'}
    return subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '--git-dir', str(repo),
                           *args], env=env, input=input, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

def index_files(repo: Path, index: Path, extra: set[str]) -> dict[str, str]:
    entries = []
    for item in git(repo, index, 'ls-files', '-s', '-z').split(b'\0'):
        if not item:
            continue
        meta, path_bytes = item.split(b'\t', 1)
        mode, oid, stage = meta.split()
        path = path_bytes.decode('utf-8')
        if path.endswith('.py') or path in extra:
            if mode not in (b'100644', b'100755') or stage != b'0':
                raise ValueError(f'Non-regular source cannot be inspected: {path}')
            entries.append((path, oid))
    raw = git(repo, index, 'cat-file', '--batch', input=b''.join(oid + b'\n' for _, oid in entries))
    offset = 0
    result = {}
    for path, oid in entries:
        end = raw.index(b'\n', offset)
        actual, kind, size = raw[offset:end].split()
        assert actual == oid and kind == b'blob'
        length = int(size)
        result[path] = raw[end + 1:end + 1 + length].decode('utf-8', errors='replace')
        offset = end + length + 2
    assert offset == len(raw)
    return result
