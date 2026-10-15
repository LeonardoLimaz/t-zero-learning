"""Compute seed ranges and a conservative operational verdict for Part 3."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ['sac-base', 'sac-reuse2', 'sac-early', 'sac-qslow']


def endpoint(records, case, metric):
    values = [records[f'{case}-s{seed}']['final_metrics'][metric] for seed in (1,2,3)]
    return dict(mean=float(np.mean(values)), minimum=float(min(values)),
                maximum=float(max(values)), spread=float(max(values)-min(values)), values=values)


def analyze():
    records = {r['run']:r for r in json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))}
    missing = [f'{c}-s{s}' for c in CASES for s in (1,2,3) if f'{c}-s{s}' not in records]
    if missing:
        raise RuntimeError(f'Part 3 incomplete: {missing}')
    groups = {}
    for case in CASES:
        groups[case] = dict(evaluation=endpoint(records,case,'eval/mean_return'),
                            train=endpoint(records,case,'charts/episodic_return_mean_last100'))
    base = groups['sac-base']
    base['verdict'] = 'referência'
    for case in CASES[1:]:
        group = groups[case]
        e, b = group['evaluation'], base['evaluation']
        difference = e['mean']-b['mean']
        larger_spread = max(e['spread'], b['spread'])
        if e['minimum'] > b['maximum'] and difference > larger_spread and group['train']['mean'] > base['train']['mean']:
            verdict = 'melhora'
        elif e['maximum'] < b['minimum'] and group['train']['mean'] < base['train']['mean']:
            verdict = 'piora'
        else:
            verdict = 'indistinguível do ruído'
        group.update(verdict=verdict, eval_difference=difference,
                     eval_overlap=max(0.0,min(e['maximum'],b['maximum'])-max(e['minimum'],b['minimum'])))
    passed = [c for c in CASES[1:] if groups[c]['verdict']=='melhora']
    final_case = max(passed, key=lambda c:groups[c]['evaluation']['mean']) if passed else 'sac-base'
    closest_case = max(CASES[1:], key=lambda c:groups[c]['evaluation']['mean'])
    result = dict(groups=groups, final_case=final_case, closest_case=closest_case,
                  criterion='Melhora: intervalos de avaliação sem sobreposição, diferença de médias maior '
                            'que a maior amplitude entre seeds e média de treino também maior. '
                            'Piora observada: faixa de avaliação inteiramente abaixo da referência '
                            'e média de treino menor. Nos demais casos, não há evidência suficiente '
                            'para separar o efeito da variação entre seeds. Critério operacional '
                            'conservador; 3 seeds não constituem teste de significância ou de equivalência.')
    (ROOT/'output'/'analise.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


if __name__=='__main__':
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
