"""Record user-requested early closure using only already completed batches."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / 'rl_v5'
REF = V5 / 'runs/structured-split'
RUN = V5 / 'runs/refined-teachers'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    ref, selection = read(REF / 'results.json'), read(REF / 'selection.json')
    batches = {}
    for path in sorted(RUN.glob('*.json')):
        data = read(path)
        if not isinstance(data, list) or not data or not isinstance(data[0], dict) or 'reward' not in data[0]:
            continue
        errors = sum(len(r.get('errors', [])) + len(r.get('lineage_errors', [])) for r in data)
        assert errors == 0, path
        batches[path.stem] = dict(games=len(data), wins=sum(r['reward'] == 1 for r in data),
            draws=sum(r['reward'] == 0 for r in data), losses=sum(r['reward'] == -1 for r in data),
            execution_and_lineage_errors=errors)
    policy = dict(architecture='gru', **selection['selected']['gru'], stochastic=False,
        checkpoint_sha256=selection['checkpoint_sha256']['gru'],
        frozen_base_sha256=ref['frozen_base_sha256']['gru'],
        validation=selection['validation']['gru']['aggressive'], test=ref['tests']['gru']['score'],
        selection_basis='Retain reference validation winner; completed refinement validation did not beat it.',
        test_protocol=dict(source=str(REF / 'results.json'), maps=['Colosseum.map', 'devil.map'],
            seeds=[9511, 9517], sides=['A', 'B'], new_refinement_tests_run=False),
        use='Local native teacher versus original scheme 1. Official sandbox deployment unverified.',
        source_manifest=str(RUN / 'manifest.json'), early_stopped=True)
    assert hashlib.sha256(Path(policy['weights']).read_bytes()).hexdigest() == policy['checkpoint_sha256']
    write(V5 / 'best-policy.json', policy)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    closure = dict(time=now, early_stopped=True, reason='User requested rapid convergence; no further matches.',
        reference_completed_games=ref['total_games'], completed_refinement_games=sum(b['games'] for b in batches.values()),
        completed_batches=batches, refinement_protocol_complete=False,
        stopped_at='GRU initial stochastic validation batch; no complete batch saved.',
        omitted=['Remaining GRU execution controls', 'GRU feeding-v2 checks', 'New independent tests with seeds 9811/9817'],
        audit_scope='Reference fully audited previously; refinement batch scores and recorded execution/lineage errors checked at closure.')
    write(RUN / 'closure.json', closure)
    write(V5 / 'pipeline-status.json', dict(stage='complete', time=now, early_stopped=True,
        experiment_protocol_complete=False, reason=closure['reason'], report=str(V5 / '优化总结.md')))
    path = V5 / '中间诊断.md'
    old = path.read_text(encoding='utf-8')
    path.write_text('# 中间诊断（历史记录）\n\n实验已按用户要求提前收束，剩余进程已停止。最终结论见 `优化总结.md`，完整参考实验见 `实验报告.md`。以下保留当时的诊断记录，不代表当前仍在运行。\n\n' + old.split('\n', 1)[1], encoding='utf-8')
    print(json.dumps(dict(reference_games=ref['total_games'], refinement_games=closure['completed_refinement_games'],
        validation=policy['validation'], test=policy['test']), ensure_ascii=False))

if __name__ == '__main__':
    main()
