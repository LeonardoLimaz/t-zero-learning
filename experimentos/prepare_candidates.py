"""Freeze one-factor candidates after observing baseline history at 10,201 steps."""
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experimentos' / 'plano.json'
cases = [
    ('sac-reuse2', 'sac_hopper_reuse2', 'updates_per_step', 2,
     'sac-base-s1, passo 10.201: q_minus_return=-107,0, retorno100=99,8. '
     'Hipótese: mais ajuste do crítico por amostra pode reduzir o atraso inicial de Q; custa tempo.'),
    ('sac-early', 'sac_hopper_early', 'learning_starts', 1000,
     'sac-base-s1: até 5.000 passos só há ações aleatórias, e aos 10.201 retorno100=99,8. '
     'Antecipar o aprendizado libera 4.000 passos adicionais de interação guiada dentro do orçamento curto.'),
    ('sac-qslow', 'sac_hopper_qslow', 'q_lr', 3e-4,
     'sac-base-s1, passo 10.201: alpha=0,105 após começar em 1; entropy=-2,50 ante alvo=-3. '
     'Diminuir q_lr testa adaptação menos rápida; este parâmetro controla tanto Q quanto alpha, '
     'então o experimento não isola o otimizador da temperatura.'),
]
plan = json.loads(PLAN.read_text(encoding='utf-8'))
if any(job['phase'] == 'candidates' for job in plan):
    raise RuntimeError('Candidate plan already exists')
registered = datetime.now(timezone.utc).isoformat()
notes = []
for case, config, key, value, reason in cases:
    data = yaml.safe_load((ROOT/'configs'/'sac_hopper.yml').read_text(encoding='utf-8'))
    old = data['sac'][key]
    data['sac'][key] = value
    (ROOT/'configs'/f'{config}.yml').write_text(
        '# Part 3: one-factor candidate relative to sac_hopper.\n'+
        yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    notes.append(dict(case=case, key='sac.'+key, old=old, new=value,
                      registered_utc=registered, reason=reason))
    plan.extend(dict(phase='candidates', case=case, config=config, seed=seed) for seed in (1,2,3))
PLAN.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
(ROOT/'experimentos'/'motivos_parte3.json').write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding='utf-8')
print('Registered 3 one-factor candidates, 3 seeds each:', registered)
