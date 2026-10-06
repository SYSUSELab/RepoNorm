"""Frozen pytest receipt parser."""
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
def check(name, paths, selector=None, isolated=False):
    xml = Path('/tmp') / f'norm-pilot-{name}.xml'
    cmd = ['python', '-m', 'pytest', *paths, '-q', f'--junitxml={xml}']
    if isolated:
        cmd += ['--confcutdir=/checks/tests', '-c', '/dev/null']
    if selector:
        cmd += ['-k', selector]
    # Same narrow old-upstream / new-pytest adjustment as v5 certification.
    cmd += ['-W', 'ignore:Passing a non-Collection iterable to parametrize is deprecated:pytest.PytestRemovedIn10Warning']
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    except subprocess.TimeoutExpired as error:
        print((error.stdout or b'').decode(errors='replace'))
        return {'status': 'error', 'reason': 'subprocess_timeout', 'exit_code': 2,
                'seconds': round(time.monotonic() - start, 3)}
    print(proc.stdout.decode(errors='replace'))
    cases = list(ET.parse(xml).iter('testcase')) if xml.exists() else []
    counts = {'tests': len(cases), 'failures': sum(c.find('failure') is not None for c in cases),
              'errors': sum(c.find('error') is not None for c in cases),
              'skipped': sum(c.find('skipped') is not None and
                             c.find('skipped').get('type') != 'pytest.xfail' for c in cases),
              'xfailed': sum(c.find('skipped') is not None and
                             c.find('skipped').get('type') == 'pytest.xfail' for c in cases)}
    status, exit_code = classify(proc.returncode, counts, isolated)
    return {'status': status, 'exit_code': exit_code, 'pytest_exit_code': proc.returncode,
            'counts': counts, 'seconds': round(time.monotonic() - start, 3), 'command': cmd}

def classify(returncode, counts, isolated=False):
    if returncode == 0 and counts['tests'] > counts['skipped'] + counts['xfailed']:
        status, exit_code = 'pass', 0
    elif returncode == 1 and counts['failures'] and not counts['errors']:
        status, exit_code = 'fail', 1
    else:
        status, exit_code = 'error', 2
    if isolated and (counts['skipped'] or counts['xfailed']):
        status, exit_code = 'error', 2
    return status, exit_code
