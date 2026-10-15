"""Compact status from runner metadata and existing progress log tail."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
active_only='--active-only' in sys.argv
plan=json.loads((ROOT/'experimentos'/'plano.json').read_text(encoding='utf-8'))
planned={f'{j["case"]}-s{j["seed"]}' for j in plan}
completed=0
active=0
for folder in sorted((ROOT / 'experimentos' / 'resultados').iterdir()):
    if folder.name not in planned:
        continue
    meta = folder / 'command.json'
    if not meta.exists():
        continue
    data = json.loads(meta.read_text(encoding='utf-8'))
    completed+=data['status']=='complete'
    active+=data['status']=='running'
    if active_only and data['status']!='running':
        continue
    with (folder / 'train.log').open('rb') as file:
        file.seek(max(0, (folder / 'train.log').stat().st_size-3500))
        tail = file.read().decode('utf-8', errors='replace')
    lines = [l for l in re.split(r'[\r\n]', tail) if l.strip()]
    last=lines[-1] if lines else ''
    if active_only:
        steps=re.search(r'(\d+)/(\d+)',last)
        reward=re.search(r'r_last100=([-+\d.]+)',last)
        print(folder.name, f'{100*int(steps[1])/int(steps[2]):.0f}%' if steps else 'inicialização/avaliação',
              'retorno100='+reward[1] if reward else '')
    else:
        print(folder.name, data['status'], last)
print(f'Concluídos: {completed}/19 previstos; em execução: {active}')
