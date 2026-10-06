"""Combine rule checks with explicit, patch-bound semantic review.

Unknown patches do not inherit a reference's document/compatibility verdict.
The human-readable review ledger is not evidence of human participation.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tool_status(checks):
    if not checks or any(c.get('exit_code') is None or c.get('status') == 'error' for c in checks):
        return 'error'
    return 'pass' if all(c['exit_code'] == 0 for c in checks) else 'fail'


def review_status(cards, digest, key):
    status = cards.get(digest, {}).get(key, {}).get('status', 'review_required')
    return status if status in ('pass', 'fail', 'error', 'review_required') else 'error'


def assess(record, data, cards):
    task = next(t for t in data['tasks'] if t['id'] == record['task'])
    selected_ids = {item['norm_id'] for item in task['norms']}
    norms = [n for n in data['norms'] if n['id'] in selected_ids]
    result = {}
    dynamic = record['dynamic']
    for norm in norms:
        kind = norm['oracle_kind']
        semantic = review_status(cards, record['patch_sha256'], kind)
        reason = None
        if kind in ('runtime', 'public_export'):
            status = dynamic.get('norms', {}).get(norm['id'], {}).get('status', 'error')
        elif kind in ('style', 'typing'):
            status = tool_status(record['static'][kind])
        elif kind == 'authored_tests':
            green, red = dynamic.get('author_green', {}), dynamic.get('author_red', {})
            if not dynamic.get('authored_paths'):
                status, reason = 'fail', 'no_authored_tests_in_submission'
            elif green.get('status') == 'error':
                status, reason = 'error', 'red_green_not_validly_observed'
            elif red.get('status') == 'error':
                # A new public API may legitimately be absent at collection
                # time on base. Require explicit attribution, not an automatic
                # pass or rejection of all such authored tests.
                attribution = review_status(cards, record['patch_sha256'], 'authored_red_failure')
                status = semantic if green.get('status') == 'pass' and attribution == 'pass' else 'review_required'
                reason = 'base_collection_failure_requires_specific_attribution'
            elif green.get('status') != 'pass' or red.get('status') != 'fail':
                status, reason = 'fail', 'authored_tests_do_not_show_red_green'
            elif not red.get('counts', {}).get('failures'):
                status, reason = 'error', 'no_asserted_or_execution_failure_on_base'
            else:
                status = semantic
                reason = 'red_green_plus_task_relevance_review'
        elif kind == 'compatibility':
            status = dynamic.get('syntax', {}).get('status', 'error')
            if status == 'pass':
                status = semantic
            reason = 'minimum_grammar_plus_API_review_not_full_runtime_matrix'
        elif kind == 'changelog' and 'CHANGES.rst' not in record['changed']:
            status, reason = 'fail', 'no_changed_changelog'
        else:
            status = semantic
            if kind == 'docs' and dynamic.get('doc_examples', {}).get('status') == 'fail':
                status, reason = 'fail', 'documented_example_failed'
        result[norm['id']] = {'status': status, 'kind': kind,
                              'strength': norm['strength'], 'reason': reason}
    statuses = {n['status'] for n in result.values()}
    overall = 'error' if 'error' in statuses else 'fail' if 'fail' in statuses else (
        'review_required' if 'review_required' in statuses else 'pass')
    return {'task': record['task'], 'state': record['state'], 'patch_sha256': record['patch_sha256'],
            'norms': result, 'overall': overall,
            'functional': dynamic.get('functional', {}).get('status', 'error'),
            'regression': dynamic.get('regression', {}).get('status', 'error')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipts', type=Path)
    parser.add_argument('--reviews', type=Path, default=HERE / 'review_cards.json')
    args = parser.parse_args()
    records = json.loads(args.receipts.read_text())
    data = json.loads((HERE / 'norms.json').read_text())
    cards = json.loads(args.reviews.read_text())['patches']
    results = [assess(r, data, cards) for r in records]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
