"""Sync measured runs, verify config/endpoints, and publish an interactive report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
ENTITY = os.environ.get('WANDB_ENTITY','leonardo-leonardo2-federal-university-of-goi-s')
PROJECT = os.environ.get('WANDB_PROJECT','sac-assignment')
SAVED = ROOT/'experimentos'/'publicacao.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sync-only', action='store_true')
    args=p.parse_args()
    import wandb
    summaries = json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))
    previous = json.loads(SAVED.read_text(encoding='utf-8')) if SAVED.exists() else {}
    urls = previous.get('runs',{})
    videos_uploaded = previous.get('videos_uploaded',{})
    code_uploaded = previous.get('code_uploaded',{})
    pending=[r for r in summaries if r['run'] not in urls]
    os.environ['WANDB_MODE']='online'
    if pending:
        with (ROOT/'experimentos'/'wandb_sync.log').open('a',encoding='utf-8') as log:
            command=[sys.executable,'-m','wandb','sync','--entity',ENTITY,'--project',PROJECT,
                     *[r['offline_path'] for r in pending]]
            print(f'Syncing {len(pending)} measured runs',flush=True)
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    api=wandb.Api(timeout=120)
    for r in summaries:
        run=api.run(f'{ENTITY}/{PROJECT}/{r["run_id"]}')
        cfg=run.config
        for key in ('exp_name','algorithm','env_id','seed','total_timesteps','network','network_kwargs'):
            if cfg[key] != r['config'][key]:
                raise RuntimeError(f'Config mismatch {r["run"]}: {key}')
        section = cfg['algorithm']
        if cfg[section] != r['config'][section]:
            raise RuntimeError(f'Algorithm hyperparameter mismatch {r["run"]}')
        for metric in ('eval/mean_return','charts/episodic_return_mean_last100'):
            if abs(float(run.summary[metric])-r['final_metrics'][metric])>1e-5:
                raise RuntimeError(f'Endpoint mismatch {r["run"]}: {metric}')
        run.name=r['run']
        run.tags=['atividade-sac',r['phase'],f'seed-{cfg["seed"]}']
        run.update()
        source=ROOT/'algorithms'/f'{section}.py'
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        command=json.loads((ROOT/'experimentos'/'resultados'/r['run']/'command.json').read_text(encoding='utf-8'))
        if command['algorithm_sha256']!=source_hash:
            raise RuntimeError(f'Code changed since training: {r["run"]}')
        if r['run'] not in code_uploaded:
            run.upload_file(str(source),root=str(source.parent))
            run.summary['assignment/source_sha256']=source_hash
            run.summary.update()
            code_uploaded[r['run']]=dict(filename=source.name,sha256=source_hash)
        if r['run'] not in videos_uploaded:
            videos = list(Path(r['run_dir']).glob('videos/*.mp4'))
            for video in videos:
                run.upload_file(str(video),root=str(video.parent))
            videos_uploaded[r['run']]=[video.name for video in videos]
        urls[r['run']]=run.url.replace(chr(92),'/')
        print('Verified',r['run'],flush=True)
    publication=dict(previous,project_url=f'https://wandb.ai/{ENTITY}/{PROJECT}',
                     fork_url='https://github.com/LeonardoLimaz/t-zero-learning',
                     runs=urls,verified_runs=len(summaries),visibility='USER_READ',
                     videos_uploaded=videos_uploaded,
                     code_uploaded=code_uploaded,
                     git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    SAVED.write_text(json.dumps(publication,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.sync_only:
        return
    if len(summaries)!=19:
        raise RuntimeError(f'Expected 19 completed runs, found {len(summaries)}')
    if publication.get('report_url'):
        print('Existing report preserved:',publication['report_url'],flush=True)
        return
    import wandb_workspaces.reports.v2 as wr
    analysis=json.loads((ROOT/'output'/'analise.json').read_text(encoding='utf-8'))
    predictions=json.loads((ROOT/'experimentos'/'previsoes.json').read_text(encoding='utf-8'))
    final=analysis['final_case']
    display=final if final!='sac-base' else analysis['closest_case']

    def grid(name,ids,metrics):
        filters=' or '.join(f'Config("exp_name") == "{i}"' for i in ids)
        panels=[wr.LinePlot(title=title,x=x,y=[metric],aggregate=False,smoothing_type='none',
                            max_runs_to_show=len(ids),layout=wr.Layout(x=0,y=n*6,w=24,h=6))
                for n,(title,metric,x) in enumerate(metrics)]
        return wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY,project=PROJECT,name=name,filters=filters)],panels=panels)

    returns='charts/episodic_return_mean_last100'
    base_ids=[f'{final}-s{s}' for s in (1,2)]
    part3=[f'{c}-s{s}' for c in ('sac-base',display) for s in (1,2,3)]
    table='| Run | Avaliação média | Treino últimos 100 | Passos efetivos | Tempo de treino (s) |\n|---|---:|---:|---:|---:|\n'
    for r in sorted(summaries,key=lambda r:r['run']):
        m=r['final_metrics']
        table+=(f'| [{r["run"]}]({urls[r["run"]]}) | {m["eval/mean_return"]:.1f} | '
                f'{m[returns]:.1f} | {r["completed_steps"]:,} | {r["train_seconds"]:.0f} |\n')
    report=wr.Report(entity=ENTITY,project=PROJECT,
                     title='Atividade SAC - Hopper: eficiência, temperatura, críticos e PPO',
                     description='Experimentos originais: 3 seeds por configuração, 2 seeds por ablação e PPO; projeto público.')
    report.blocks=[
        wr.H1(text='Protocolo e rastreabilidade'),
        wr.MarkdownBlock(text='Leonardo Lima de Oliveira. Hopper-v5, 100.000 passos planejados por treino; '
                         'o PPO fornecido executa 98.304 por completar apenas rollouts inteiros. '
                         'Pendulum de sanidade: 20.000 passos. Curvas originais do W&B. '
                         '[Fork com código e configs](https://github.com/LeonardoLimaz/t-zero-learning).'),
        wr.H1(text='Parte 3 - Candidatos e variação entre seeds'),
        wr.MarkdownBlock(text=f'Config selecionado: **{final}**. '+analysis['criterion']),
        grid('Inicial e final (ou candidato mais próximo)',part3,[
            ('Retorno x passos',returns,'global_step'),
            ('Temperatura','losses/alpha','global_step'),
            ('Entropia','losses/entropy','global_step')]),
        wr.H1(text='Parte 4 - Temperatura fixa'),
        wr.MarkdownBlock(text='Previsão registrada em '+predictions['registered_utc']+': '+predictions['fixed']+
                         f' Alpha fixo={predictions["alpha_choice"]["value"]}, mediana da cauda dos 3 seeds selecionados.'),
        grid('Autotune x alpha fixo',base_ids+[f'sac-fixed-s{s}' for s in (1,2)],[
            ('Retorno x passos',returns,'global_step'),('Alpha','losses/alpha','global_step'),
            ('Entropia','losses/entropy','global_step'),('Alvo de entropia','charts/target_entropy','global_step')]),
        wr.H1(text='Parte 4 - Um crítico no alvo e no ator'),
        wr.MarkdownBlock(text='Previsão: '+predictions['single']+' A implementação mantém Q2 e sua perda; '
                         'a ablação remove o mínimo conservador, não metade da computação.'),
        grid('Mínimo de dois Q x Q1',base_ids+[f'sac-single-s{s}' for s in (1,2)],[
            ('Retorno x passos',returns,'global_step'),('Q menos retorno soft','charts/q_minus_return','global_step'),
            ('Q1 médio','losses/qf1_values','global_step')]),
        wr.H1(text='Parte 5 - SAC x PPO'),
        wr.MarkdownBlock(text='Previsão: '+predictions['comparison']),
        grid('SAC selecionado x PPO fornecido',base_ids+[f'ppo-hopper-s{s}' for s in (1,2)],[
            ('Retorno x passos',returns,'global_step'),('Retorno x tempo relativo',returns,'_runtime'),
            ('Passos por segundo','charts/SPS','global_step'),('Atualizações','charts/num_updates','global_step')]),
        wr.H1(text='Todos os resultados e runs'),wr.MarkdownBlock(text=table)
    ]
    report.save()
    publication['report_url']=report.url.replace(chr(92),'/')
    SAVED.write_text(json.dumps(publication,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Report:',publication['report_url'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
