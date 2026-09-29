"""Wait for the active reference process, then run serial refinements and audits.

This coordinator never launches concurrent matches and never treats an incomplete
reference batch as successful. It is resumable through experiment fingerprints.
"""
from pathlib import Path
import argparse, ctypes, json, subprocess, sys, time, shutil

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv-train/Scripts/python.exe'
REFERENCE = ROOT / 'rl_v5/runs/structured-split'
REFINED = ROOT / 'rl_v5/runs/refined-teachers'
STATUS = ROOT / 'rl_v5/pipeline-status.json'

def status(stage, **extra):
    data = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(STATUS)
    print(json.dumps(data), flush=True)

def command(args):
    print('RUN', *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def audit(directory):
    node = shutil.which('node')
    assert node, 'Node runtime required for official replay decoding'
    command([node, ROOT/'rl_v5/diagnostics/replay_metrics.cjs', directory/'episodes'])
    command([PYTHON, ROOT/'rl_v5/diagnostics/analyze.py', directory])
    command([PYTHON, ROOT/'rl_v5/diagnostics/check_birth_features.py', directory])
    analysis = json.loads((directory/'analysis.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    assert analysis['complete'] and analysis['completed_games'] == results['total_games']
    assert all('final_longest' in row for row in analysis['records'].values())
    birth = json.loads((directory/'birth-feature-audit.json').read_text())
    assert birth.get('complete_body_checked', 0) == birth.get('complete_body_matches', 0)
    assert birth.get('cut_checked', 0) == birth.get('cut_matches', 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-pid', type=int)
    args = parser.parse_args()
    status('waiting-reference')
    if args.reference_pid:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, args.reference_pid)
        if not handle:
            assert (REFERENCE/'results.json').exists(), f'Cannot observe reference PID: {ctypes.get_last_error()}'
        else:
            try:
                while True:
                    state = kernel.WaitForSingleObject(handle, 0)
                    if state == 0: break
                    assert state == 258, f'Process wait failed: {state}'
                    time.sleep(30)
            finally:
                kernel.CloseHandle(handle)
    assert (REFERENCE/'results.json').exists(), 'Reference process ended without complete results'
    status('auditing-reference')
    audit(REFERENCE)
    command([PYTHON, ROOT/'rl_v5/diagnostics/report.py'])
    status('refining')
    with (ROOT/'rl_v5/refinement.log').open('a', encoding='utf-8') as log:
        subprocess.run([str(PYTHON), '-u', str(ROOT/'rl_v5/refinements/followup_experiment.py')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    status('auditing-refinement')
    audit(REFINED)
    command([PYTHON, ROOT/'rl_v5/diagnostics/refinement_report.py'])
    status('complete', report=str(ROOT/'rl_v5/优化总结.md'))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc))
        raise
