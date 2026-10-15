"""Advance the dependent phases as soon as all Part 3 seeds finish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(TMP=str(ROOT),TEMP=str(ROOT),PYTHONIOENCODING='utf-8',PYTHONWARNINGS='ignore')
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
required=[f'{case}-s{seed}' for case in ('sac-base','sac-reuse2','sac-early','sac-qslow') for seed in (1,2,3)]
print('Waiting for all 12 Part 3 runs; next phases are prepared automatically.',flush=True)
while True:
    states=[]
    for name in required:
        path=ROOT/'experimentos'/'resultados'/name/'command.json'
        if not path.exists():
            states.append('queued')
            continue
        try:
            states.append(json.loads(path.read_text(encoding='utf-8'))['status'])
        except json.JSONDecodeError:
            states.append('running')
    if 'failed' in states:
        raise RuntimeError('A required Part 3 run failed; inspect logs before continuing.')
    if all(s=='complete' for s in states):
        break
    time.sleep(15)


def execute(script,*args):
    subprocess.run([sys.executable,str(ROOT/'experimentos'/script),*args],cwd=ROOT,check=True)


execute('export_results.py')
if not (ROOT/'experimentos'/'previsoes.json').exists():
    execute('prepare_followups.py')
print('Part 3 complete; predictions frozen; starting 4 ablation runs and 2 PPO runs.',flush=True)
execute('run_suite.py','--phase','followups','--workers','6')
execute('export_results.py')
execute('report_numbers.py')
print('ALL 19 RUNS COMPLETE AND EXPORTED',flush=True)
