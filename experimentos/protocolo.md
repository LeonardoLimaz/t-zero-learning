# Atividade SAC - protocolo e reprodução

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

Config final medido: **sac-base**, reproduzido em `configs/sac_hopper_final.yml`.

Melhora: intervalos de avaliação sem sobreposição, diferença de médias maior que a maior amplitude entre seeds e média de treino também maior. Piora observada: faixa de avaliação inteiramente abaixo da referência e média de treino menor. Nos demais casos, não há evidência suficiente para separar o efeito da variação entre seeds. Critério operacional conservador; 3 seeds não constituem teste de significância ou de equivalência.

As razões dos candidatos foram registradas em `motivos_parte3.json` após observar o padrão aos 10.201 passos. As previsões de ablação e comparação foram congeladas em `previsoes.json` em 2026-10-15T15:02:40.874977+00:00, antes de iniciar esses treinos.

## Dados publicados

[Projeto W&B público](https://wandb.ai/leonardo-leonardo2-federal-university-of-goi-s/sac-assignment). `resultados_resumo.csv` liga cada um dos 19 runs aos seus resultados medidos e URL. Os históricos completos, configs salvos e vídeos estão no projeto W&B.
