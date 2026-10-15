"""Choose the Part 3 config and freeze predictions BEFORE Parts 4/5 runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import yaml
from analyze import analyze

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT/'experimentos'/'previsoes.json'
if PREDICTIONS.exists():
    raise RuntimeError('Predictions already registered; preserve the original record')
selection = analyze()
case = selection['final_case']
records = {r['run']:r for r in json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))}
data = records[f'{case}-s1']['config'].copy()
runtime_keys = ['batch_size','minibatch_size','num_iterations','env_kwargs_id','exp_name']
for key in runtime_keys:
    data.pop(key, None)
data['seed'] = 1
alpha_values = []
for seed in (1,2,3):
    rows = [json.loads(line) for line in (ROOT/'experimentos'/'resultados'/f'{case}-s{seed}'/'wandb_history.jsonl').read_text(encoding='utf-8').splitlines()]
    alpha_values.extend(row['losses/alpha'] for row in rows
                        if row.get('global_step',0)>=80000 and 'losses/alpha' in row)
alpha_median = float(np.median(alpha_values))
fixed_alpha = float(f'{alpha_median:.2g}')
registered = datetime.now(timezone.utc).isoformat()
predictions = dict(registered_utc=registered, final_case=case,
                  alpha_choice=dict(value=fixed_alpha, tail_median=alpha_median,
                                    tail_min=float(min(alpha_values)),tail_max=float(max(alpha_values)),
                                    source_steps='80.000-100.000, seeds 1/2/3'),
                  fixed='Com alpha fixo próximo da mediana final do SAC escolhido, espero retorno '
                        'parecido ao final, mas maior desvio de entropia em relação a -3 durante '
                        'o aprendizado; sem autotune o coeficiente não reage à política.',
                  single='Ao substituir min(Q1,Q2) por Q1, espero q_minus_return mais positivo '
                         'e maior instabilidade/menor retorno. Essa previsão pode falhar no orçamento '
                         'curto: Q inicialmente subestima, e a política visitada também muda.',
                  comparison='Espero que o SAC alcance maior retorno por passo devido ao replay e '
                             'às atualizações frequentes, mas que o PPO processe passos muito mais '
                             'rápido. No intervalo de tempo comum, o PPO pode estar à frente. '
                             'Usarei as seeds 1/2 já medidas do SAC escolhido e duas novas seeds '
                             'do PPO com o comando sem ajuste do enunciado.')
PREDICTIONS.write_text(json.dumps(predictions,indent=2,ensure_ascii=False),encoding='utf-8')
for name, edits in [('sac_hopper_final', {}), ('sac_hopper_fixed_alpha', {'autotune':False,'alpha':fixed_alpha}),
                    ('sac_hopper_single_q', {'clipped_double_q':False})]:
    config = json.loads(json.dumps(data))
    config['sac'].update(edits)
    (ROOT/'configs'/f'{name}.yml').write_text('# Frozen from Part 3: '+case+'\n'+
                                           yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
ppo = yaml.safe_load((ROOT/'configs'/'ppo_pendulum.yml').read_text(encoding='utf-8'))
ppo.update(env_id='Hopper-v5',total_timesteps=100000,track=True)
ppo['ppo_continuous_action']['gamma']=0.99
(ROOT/'configs'/'ppo_hopper_sac_comparison.yml').write_text(
    '# Exact Part 5 CLI overrides applied to ppo_pendulum.\n'+yaml.safe_dump(ppo,sort_keys=False),encoding='utf-8')
plan_path = ROOT/'experimentos'/'plano.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
for case_name,config in [('sac-fixed','sac_hopper_fixed_alpha'),('sac-single','sac_hopper_single_q')]:
    plan.extend(dict(phase='followups',case=case_name,config=config,seed=seed) for seed in (1,2))
plan.extend(dict(phase='followups',case='ppo-hopper',config='ppo_pendulum',seed=seed,
                 overrides=['env_id=Hopper-v5','total_timesteps=100000',
                            'ppo_continuous_action.gamma=0.99','track=true']) for seed in (1,2))
plan_path.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(predictions,indent=2,ensure_ascii=False))
