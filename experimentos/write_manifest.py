"""Write a portable run index and protocol, without machine-specific paths."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))
publication=json.loads((ROOT/'experimentos'/'publicacao.json').read_text(encoding='utf-8'))
analysis=json.loads((ROOT/'output'/'analise.json').read_text(encoding='utf-8'))
predictions=json.loads((ROOT/'experimentos'/'previsoes.json').read_text(encoding='utf-8'))
if len(records)!=19:
    raise RuntimeError('The run index must contain all 19 completed runs')
fields=['run','phase','algorithm','environment','seed','steps','evaluation_mean',
        'training_last100','train_seconds','sps','num_updates','wandb_url']
with (ROOT/'experimentos'/'resultados_resumo.csv').open('w',newline='',encoding='utf-8') as file:
    writer=csv.DictWriter(file,fieldnames=fields)
    writer.writeheader()
    for r in sorted(records,key=lambda r:r['run']):
        c=r['config'];m=r['final_metrics']
        writer.writerow(dict(run=r['run'],phase=r['phase'],algorithm=c['algorithm'],environment=c['env_id'],
                             seed=c['seed'],steps=r['completed_steps'],evaluation_mean=m['eval/mean_return'],
                             training_last100=m['charts/episodic_return_mean_last100'],train_seconds=r['train_seconds'],
                             sps=m['charts/SPS'],num_updates=m['charts/num_updates'],wandb_url=publication['runs'][r['run']]))

text='''# Atividade SAC - protocolo e reprodução

Autor: Leonardo Lima de Oliveira. Trabalho individual.

O código de `algorithms/sac.py` preenche somente os dois blocos da atividade.
`train.py` foi corrigido para mostrar as taxas de SAC (`policy_lr` e `q_lr`),
pois a mensagem original acessava `learning_rate`, inexistente nesse algoritmo.

## Protocolo

- SAC Hopper: exatamente 100.000 passos por run, CPU, uma thread por processo.
- Referência e três candidatos: seeds 1, 2 e 3. Cada candidato muda uma chave.
- Ablações e PPO: seeds 1 e 2. O SAC escolhido da Parte 3 é reutilizado na comparação.
- O PPO usa exatamente o comando pedido (com nome de experimento próprio); seu
  laço só conclui rollouts inteiros, executando 98.304 dos 100.000 passos planejados.
- Vídeo da avaliação determinística do SAC e checkpoints foram mantidos como no YAML fornecido.
- Até seis treinos simultâneos. Tempos refletem execução concorrente, não medições isoladas de hardware.
- Avaliação SAC: ação média, 10 episódios, além de uma avaliação amostrada de 10 episódios.
  O PPO mantém a avaliação amostrada e os wrappers de normalização fornecidos.

## Executar a suíte fixa

Instale as dependências do repositório num ambiente Python com MuJoCo funcionando.
Configure `WANDB_PROJECT=sac-assignment` e seu `WANDB_ENTITY` no `.env` ou no ambiente.
O runner grava W&B offline e preserva os logs originais; use `wandb sync` para publicar.
Em Windows, a renderização MuJoCo padrão foi suficiente; em Linux sem tela, use `MUJOCO_GL=egl`.

```powershell
python -m pytest tests/test_sac.py
python experimentos/run_suite.py --phase sanity --workers 1
python experimentos/run_suite.py --phase baseline --workers 3
python experimentos/run_suite.py --phase candidates --workers 6
python experimentos/run_suite.py --phase followups --workers 6
python experimentos/export_results.py
python experimentos/analyze.py
python experimentos/report_numbers.py
python experimentos/plot_results.py
```

`plano.json` contém todos os nomes, seeds, configs e overrides. Os scripts usam
o Python com que foram iniciados; não dependem do caminho de uma venv específica.
O runner evita repetir um run concluído e pede inspeção de tentativas interrompidas.

As bibliotecas de figuras e PDF estão em `requirements-analysis.txt`.
Para sincronizar e criar o relatório W&B com a SDK usada nesta atividade, instale
`requirements-publish.txt` em uma venv separada, execute `wandb login` nela e rode
`python experimentos/publish_wandb.py`. A verificação sem credenciais é
`python experimentos/verify_public.py`. Com os dados exportados, figuras e publicação
disponíveis, `python experimentos/build_report.py` gera o PDF; os textos interpretativos
em `interpretacao.json` registram as conclusões desta execução e devem ser revistos
ao analisar resultados novos.

## Escolha e previsões

'''
text+=f"Config final medido: **{analysis['final_case']}**, reproduzido em `configs/sac_hopper_final.yml`.\n\n"
text+=analysis['criterion']+'\n\n'
text+='As razões dos candidatos foram registradas em `motivos_parte3.json` após observar o padrão aos 10.201 passos. '
text+=f"As previsões de ablação e comparação foram congeladas em `previsoes.json` em {predictions['registered_utc']}, antes de iniciar esses treinos.\n\n"
text+='## Dados publicados\n\n'
text+=f"[Projeto W&B público]({publication['project_url']}). `resultados_resumo.csv` liga cada um dos 19 runs aos seus resultados medidos e URL. "
text+='Os históricos completos, configs salvos e vídeos estão no projeto W&B.\n'
(ROOT/'experimentos'/'protocolo.md').write_text(text,encoding='utf-8')
print('Portable protocol and 19-row result index written.')
