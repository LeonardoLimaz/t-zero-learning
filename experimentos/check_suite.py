"""Audit required seeds, budgets, single-factor changes and prediction chronology."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
records={r['run']:r for r in json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))}
plan=json.loads((ROOT/'experimentos'/'plano.json').read_text(encoding='utf-8'))
expected={f'{j["case"]}-s{j["seed"]}' for j in plan}
assert len(expected)==19 and set(records)==expected, 'Missing or unexpected measured runs'
prediction=json.loads((ROOT/'experimentos'/'previsoes.json').read_text(encoding='utf-8'))
predicted_at=datetime.fromisoformat(prediction['registered_utc'])
for name,r in records.items():
    c=r['config']
    assert r['completed_steps']==(98304 if c['algorithm']=='ppo_continuous_action' else c['total_timesteps']),name
    assert (Path(r['run_dir'])/'model.pt').exists(),name
    command=json.loads((ROOT/'experimentos'/'resultados'/name/'command.json').read_text(encoding='utf-8'))
    assert command['status']=='complete' and command['exit_code']==0,name
    source_hash=hashlib.sha256((ROOT/'algorithms'/f'{c["algorithm"]}.py').read_bytes()).hexdigest()
    assert command['algorithm_sha256']==source_hash,name
    if r['phase']=='followups':
        assert predicted_at<datetime.fromisoformat(command['started_utc']),name
    assert abs(np.mean(r['eval_returns'][:10])-r['final_metrics']['eval/mean_return'])<1e-5,name
    if c['algorithm']=='sac':
        assert abs(np.mean(r['eval_returns'][10:])-r['final_metrics']['eval/mean_return_sampled'])<1e-5,name
        if r['phase']!='sanity':
            assert c['env_id']=='Hopper-v5' and c['total_timesteps']==100000,name
base=records['sac-base-s1']['config']
for case,changed in [('sac-reuse2',{'updates_per_step'}),('sac-early',{'learning_starts'}),('sac-qslow',{'q_lr'})]:
    for seed in (1,2,3):
        c=records[f'{case}-s{seed}']['config']
        differences={k for k in base['sac'] if c['sac'][k]!=base['sac'][k]}
        assert differences==changed,(case,differences)
        assert c['network_kwargs']==base['network_kwargs']
        assert c['capture_video']==base['capture_video'] and c['checkpoint_every']==base['checkpoint_every']
control=records[prediction['final_case']+'-s1']['config']
for case,changed in [('sac-fixed',{'autotune','alpha'}),('sac-single',{'clipped_double_q'})]:
    for seed in (1,2):
        c=records[f'{case}-s{seed}']['config']
        differences={k for k in control['sac'] if c['sac'][k]!=control['sac'][k]}
        assert differences==changed,(case,differences)
        assert c['network_kwargs']==control['network_kwargs']
ppo=yaml.safe_load((ROOT/'configs'/'ppo_pendulum.yml').read_text(encoding='utf-8'))
for seed in (1,2):
    c=records[f'ppo-hopper-s{seed}']['config']
    assert c['env_id']=='Hopper-v5' and c['total_timesteps']==100000 and c['track']
    assert c['network_kwargs']==ppo['network_kwargs'] and c['num_envs']==ppo['num_envs']
    for key,value in ppo['ppo_continuous_action'].items():
        assert c['ppo_continuous_action'][key]==(0.99 if key=='gamma' else value),key
print('AUDIT PASSED: 19 runs; all seeds/budgets/config changes, evaluation metrics and prior predictions match.')
