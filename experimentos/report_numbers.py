"""Supplementary diagnostics for interpreting the completed experiment suite."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
records={r['run']:r for r in json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))}
result={}
for name,record in records.items():
    rows=[json.loads(line) for line in (ROOT/'experimentos'/'resultados'/name/'wandb_history.jsonl').read_text(encoding='utf-8').splitlines()]
    tail=[r for r in rows if r.get('global_step',0)>=80000]
    metrics={}
    for key in ['losses/alpha','losses/entropy','charts/q_minus_return','losses/qf1_values',
                'losses/qf2_values','losses/qf1_loss','losses/qf2_loss']:
        values=[r[key] for r in tail if key in r]
        if values:
            metrics[key]=dict(mean=float(np.mean(values)),minimum=float(min(values)),maximum=float(max(values)),n=len(values))
    entropy=[abs(r['losses/entropy']-r['charts/target_entropy']) for r in tail
             if 'losses/entropy' in r and 'charts/target_entropy' in r]
    if entropy:
        metrics['abs_entropy_error_mean']=float(np.mean(entropy))
    biases=[r['charts/q_minus_return'] for r in tail if 'charts/q_minus_return' in r]
    if biases:
        metrics['q_bias_fraction_positive']=float(np.mean(np.array(biases)>0))
    full_errors=[abs(r['losses/entropy']-r['charts/target_entropy']) for r in rows
                 if 'losses/entropy' in r and 'charts/target_entropy' in r]
    full_biases=[r['charts/q_minus_return'] for r in rows if 'charts/q_minus_return' in r]
    cutoff=[r for r in rows if 'charts/episodic_return_mean_last100' in r
            and 'global_step' in r and r['global_step']<=98304]
    matched=cutoff[-1]['charts/episodic_return_mean_last100'] if cutoff and record['completed_steps']>=98304 else None
    result[name]=dict(tail=metrics,final=record['final_metrics'],train_seconds=record['train_seconds'],
                      train_return_at_98304=matched,
                      entropy_error_full=float(np.mean(full_errors)) if full_errors else None,
                      q_bias_full=float(np.mean(full_biases)) if full_biases else None)
(ROOT/'output'/'diagnosticos.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
for name,r in result.items():
    m=r['final']; t=r['tail']
    print(name,'eval',round(m['eval/mean_return'],1),'train',round(m['charts/episodic_return_mean_last100'],1),
          'q_bias_tail',round(t.get('charts/q_minus_return',{}).get('mean',float('nan')),1),
          'entropy_error_tail',round(t.get('abs_entropy_error_mean',float('nan')),2))
