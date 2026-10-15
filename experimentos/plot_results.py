"""Publication figures from measured rows with explicit run identities."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'figuras'
RESULTS=ROOT/'experimentos'/'resultados'
RET='charts/episodic_return_mean_last100'
COLORS={'sac-base':'#246A96','sac-reuse2':'#C77928','sac-early':'#269482',
        'sac-qslow':'#9367AA','sac-fixed':'#D8872F','sac-single':'#C35F73','ppo-hopper':'#D8872F'}
STYLES={1:'-',2:'--',3:':'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,
                     'axes.labelsize':7.5,'legend.fontsize':6.2,'axes.spines.top':False,
                     'axes.spines.right':False,'grid.color':'#DDE6EC','grid.alpha':0.8,
                     'axes.grid':True,'savefig.facecolor':'white'})


def history(run):
    return [json.loads(line) for line in (RESULTS/run/'wandb_history.jsonl').read_text(encoding='utf-8').splitlines()]


def points(run,key,time=False):
    rows=[r for r in history(run) if key in r and 'global_step' in r]
    return ([r['_runtime']/60 if time else r['global_step']/1000 for r in rows],
            [r[key] for r in rows])


def curve(ax,run,key=RET,time=False,**kwargs):
    case,seed=run.rsplit('-s',1)
    x,y=points(run,key,time)
    ax.plot(x,y,color=COLORS[case],ls=STYLES[int(seed)],lw=1.1,label=run,**kwargs)
    ax.set_xlabel('Tempo relativo W&B (min)' if time else 'Passos de ambiente (mil)')


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    analysis=json.loads((ROOT/'output'/'analise.json').read_text(encoding='utf-8'))
    final=analysis['final_case']
    shown=final if final!='sac-base' else analysis['closest_case']
    # Reuse one consistent blue for the selected control across Parts 4 and 5.
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for case in ('sac-base',shown):
        for seed in (1,2,3):
            curve(axes[0],f'{case}-s{seed}')
    axes[0].set(title='Retorno durante o treino',ylabel='Retorno (últimos 100 episódios)')
    axes[0].legend(ncol=2)
    markers=['o','^','s']
    for n,(case,g) in enumerate(analysis['groups'].items()):
        e=g['evaluation']
        axes[1].plot([e['minimum'],e['maximum']],[n,n],c=COLORS[case],lw=2)
        for seed,value in enumerate(e['values'],1):
            axes[1].scatter(value,n+(seed-2)*0.08,marker=markers[seed-1],c=COLORS[case],s=28,
                            label=f'seed {seed}' if n==0 else None,zorder=3)
        axes[1].scatter(e['mean'],n,marker='|',c='#111111',s=180,zorder=4)
    axes[1].set_yticks(range(4),[f'{c}\n(s1/s2/s3)' for c in analysis['groups']])
    axes[1].invert_yaxis()
    axes[1].set(title='Avaliação: seeds e faixa mín-máx',xlabel='Retorno médio em 10 episódios')
    axes[1].legend(loc='best',ncol=3)
    save(fig,'parte3')
    if '--part3-only' in sys.argv:
        print('Part 3 figure saved:',OUT)
        return

    COLORS[final]='#246A96'
    control=[f'{final}-s{s}' for s in (1,2)]
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.2),layout='constrained')
    for run in control+[f'sac-fixed-s{s}' for s in (1,2)]:
        for ax,key in zip(axes[0],[RET,'losses/alpha','losses/entropy']):
            curve(ax,run,key)
    axes[0,0].set(title='Alpha fixo: retorno',ylabel='Retorno (últimos 100)')
    axes[0,0].legend()
    axes[0,1].set(title='Temperatura',ylabel='Alpha (escala log)',yscale='log')
    axes[0,2].set(title='Entropia da política',ylabel='- média de log π')
    axes[0,2].axhline(-3,color='#222222',ls=':',lw=1,label='alvo = -3')
    axes[0,2].legend()
    for run in control+[f'sac-single-s{s}' for s in (1,2)]:
        for ax,key in zip(axes[1],[RET,'charts/q_minus_return','losses/qf1_values']):
            curve(ax,run,key,alpha=0.85)
    axes[1,0].set(title='Q1 no alvo/ator: retorno',ylabel='Retorno (últimos 100)')
    axes[1,0].legend()
    axes[1,1].set(title='Q menos retorno soft',ylabel='Q - retorno soft')
    axes[1,1].axhline(0,color='#222222',ls=':',lw=1)
    axes[1,2].set(title='Valor previsto por Q1',ylabel='Q1 médio (replay)')
    save(fig,'parte4')

    fig,axes=plt.subplots(2,2,figsize=(7.2,4.15),layout='constrained')
    for run in control+[f'ppo-hopper-s{s}' for s in (1,2)]:
        curve(axes[0,0],run)
        curve(axes[0,1],run,time=True)
        curve(axes[1,0],run,'charts/SPS')
        curve(axes[1,1],run,'charts/num_updates')
    axes[0,0].set(title='Eficiência amostral',ylabel='Retorno (últimos 100 episódios)')
    axes[0,0].legend()
    axes[0,1].set(title='Retorno por tempo de relógio',ylabel='Retorno (últimos 100 episódios)')
    axes[1,0].set(title='Velocidade de interação e treino',ylabel='SPS acumulado (passos/s)')
    axes[1,1].set(title='Atualizações registradas',ylabel='Passos de otimização')
    save(fig,'parte5')
    print('Figures saved:',OUT)


if __name__=='__main__':
    main()
