"""Compact diagnostics from original W&B records, including live runs."""
import argparse
import json
import numpy as np
from export_results import read_history, RESULTS

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+')
args = parser.parse_args()
for name in args.runs:
    paths = list((RESULTS / name / 'wandb').glob('offline-run-*/*.wandb'))
    if len(paths) != 1:
        print(name, 'no unique history yet')
        continue
    rows = read_history(paths[0])
    keys = ['global_step', 'charts/episodic_return_mean_last100', 'losses/alpha',
            'losses/entropy', 'charts/q_minus_return', 'losses/qf1_values',
            'losses/qf1_loss', 'charts/SPS', 'charts/num_updates']
    last = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        if values:
            last[key] = {'last': values[-1], 'tail_mean': float(np.mean(values[-100:]))}
    print(name, json.dumps(last))
