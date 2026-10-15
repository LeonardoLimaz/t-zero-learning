"""Four-page report; every numerical result comes from the original W&B rows."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'
FONT=Path(matplotlib.get_data_path())/'fonts'/'ttf'
pdfmetrics.registerFont(TTFont('DejaVu',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold')
W,H=A4
MARGIN=35
WIDTH=W-2*MARGIN
INK=colors.HexColor('#163448')
BLUE=colors.HexColor('#247592')
BODY=ParagraphStyle('body',fontName='DejaVu',fontSize=8.8,leading=11.5,textColor=INK)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=7.5,leading=9.8)
CELL=ParagraphStyle('cell',parent=BODY,fontSize=7.1,leading=9.1)
RET='charts/episodic_return_mean_last100'


def dec(value,n=1):
    return f'{value:.{n}f}'.replace('.',',')


def interval(result):
    return f"{dec(result['mean'])}<br/>({dec(result['minimum'])} - {dec(result['maximum'])})"


def history(run):
    return [json.loads(l) for l in (ROOT/'experimentos'/'resultados'/run/'wandb_history.jsonl').read_text(encoding='utf-8').splitlines()]


def value_at(run,seconds):
    rows=[r for r in history(run) if RET in r and r.get('_runtime',float('inf'))<=seconds]
    return rows[-1][RET] if rows else float('nan')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records={r['run']:r for r in json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))}
    analysis=json.loads((ROOT/'output'/'analise.json').read_text(encoding='utf-8'))
    predictions=json.loads((ROOT/'experimentos'/'previsoes.json').read_text(encoding='utf-8'))
    reasons={r['case']:r for r in json.loads((ROOT/'experimentos'/'motivos_parte3.json').read_text(encoding='utf-8'))}
    prose=json.loads((ROOT/'experimentos'/'interpretacao.json').read_text(encoding='utf-8'))
    publication=json.loads((ROOT/'experimentos'/'publicacao.json').read_text(encoding='utf-8'))
    environment=json.loads((ROOT/'experimentos'/'ambiente.json').read_text(encoding='utf-8'))
    final=analysis['final_case']
    commit=publication['git_commit']
    source_url=publication['fork_url']+'/blob/'+commit+'/algorithms/sac.py'
    source=(ROOT/'algorithms'/'sac.py').read_text(encoding='utf-8').splitlines()

    def ref(pattern):
        number=next(i for i,line in enumerate(source,1) if pattern in line)
        return f'<link href="{source_url}#L{number}" color="#247592">sac.py:{number}</link>'

    c=canvas.Canvas(str(OUT/'relatorio_sac.pdf'),pagesize=A4)
    c.setAuthor('Leonardo Lima de Oliveira')
    c.setTitle('Soft Actor-Critic no Hopper: eficiência, ablações e comparação com PPO')

    def para(text,y,small=False,gap=5):
        block=Paragraph(text,SMALL if small else BODY)
        _,height=block.wrap(WIDTH,H)
        block.drawOn(c,MARGIN,y-height)
        return y-height-gap

    def heading(text,y):
        c.setFillColor(INK)
        c.setFont('DejaVu-Bold',11)
        c.drawString(MARGIN,y-11,text)
        return y-26

    def header(title):
        c.setFillColor(BLUE)
        c.setFont('DejaVu-Bold',9)
        c.drawString(MARGIN,H-32,'SAC / HOPPER-v5')
        c.setFont('DejaVu',7.2)
        c.drawRightString(W-MARGIN,H-32,'LEONARDO LIMA DE OLIVEIRA | 15/10/2026')
        c.setFillColor(INK)
        c.setFont('DejaVu-Bold',14)
        c.drawString(MARGIN,H-61,title)
        return H-77

    def image(name,y,width=WIDTH):
        path=ROOT/'output'/'figuras'/f'{name}.png'
        with Image.open(path) as im:
            height=width*im.height/im.width
        c.drawImage(str(path),(W-width)/2,y-height,width=width,height=height)
        return y-height-8

    def table(rows,widths,y,padding=4):
        body=[[Paragraph(str(v),CELL) for v in row] for row in rows]
        t=Table(body,colWidths=widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EFF3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F9FB')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),padding),
            ('BOTTOMPADDING',(0,0),(-1,-1),padding),
            ('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#BDD2DD'))]))
        _,height=t.wrap(WIDTH,H)
        t.drawOn(c,MARGIN,y-height)
        return y-height-10

    def footer(number,y):
        if y<45:
            raise RuntimeError(f'Page {number} overflow: {y}')
        c.setStrokeColor(colors.HexColor('#BFD1DB'))
        c.line(MARGIN,30,W-MARGIN,30)
        c.setFont('DejaVu',8)
        c.setFillColor(INK)
        c.drawString(MARGIN,17,'Medições próprias; históricos W&B e configurações originais preservados')
        c.drawRightString(W-MARGIN,17,f'{number} / 4')
        c.showPage()

    y=header('Parte 3 - Eficiência amostral e variação entre seeds')
    sanity=records['sac-pendulum-s1']['final_metrics']['eval/mean_return']
    y=para(f'<b>Protocolo individual.</b> Dois blocos do SAC implementados; 16 testes específicos aprovados. '
           f'Sanidade no Pendulum: avaliação determinística de {dec(sanity)}, acima de -200, com 20.000 passos. '
           'No Hopper, quatro configurações, três seeds cada, exatamente 100.000 passos por treino; '
           'avaliação com ação média em 10 episódios e retorno de treino no último log da janela de 100 episódios.',y)
    y=image('parte3',y)
    shown=final if final!='sac-base' else analysis['closest_case']
    y=para(f'<b>Figura 1.</b> À esquerda, curvas originais de sac-base e {shown}, três seeds cada. '
           'À direita, cada ponto é a média de avaliação de uma seed; linha = mín-máx e traço preto = média das seeds. '
           'Os nomes completos dos runs são o prefixo indicado mais -s1, -s2 ou -s3.',y,small=True,gap=15)
    y=heading('Uma mudança por candidato; resultados como média (mín-máx)',y)
    rows=[['<b>Configuração / alteração</b>','<b>Motivo observado antes do treino</b>',
           '<b>Avaliação final</b>','<b>Treino final</b>','<b>Veredito</b>']]
    for case,group in analysis['groups'].items():
        if case=='sac-base':
            change='sac-base<br/>Config inicial'
            reason='Referência: YAML sac_hopper fornecido, sem ajustes.'
        else:
            r=reasons[case]
            change=f"{case}<br/>{r['key']}<br/>{r['old']} → {r['new']}"
            reason=r['reason']
        rows.append([escape(change).replace('&lt;br/&gt;','<br/>'),escape(reason),
                     interval(group['evaluation']),interval(group['train']),group['verdict']])
    y=table(rows,[WIDTH*v for v in (.18,.34,.17,.17,.14)],y)
    y=para('<b>Regra de decisão.</b> '+escape(analysis['criterion']),y,small=True)
    y=para('<b>Escolha final.</b> '+prose['part3'],y)
    footer(1,y)

    y=header('Parte 4 - Temperatura e mínimo dos dois críticos')
    registered=datetime.fromisoformat(predictions['registered_utc'])-timedelta(hours=3)
    y=para(f'<b>Previsões registradas antes dos treinos:</b> {registered.strftime("%d/%m/%Y %H:%M:%S")} '
           f'(Brasília). Controle: {final}, seeds 1/2; cada ablação usa duas seeds e 100.000 passos.',y)
    y=image('parte4',y)
    y=para('<b>Figura 2.</b> Linha superior: controle x sac-fixed-s1/s2. Linha inferior: controle x '
           'sac-single-s1/s2. Mesmas cores e estilos identificam o mesmo run nos painéis da linha; '
           'curvas sem suavização adicional. O alvo de entropia é -3 (entropia diferencial pode ser negativa).',y,small=True,gap=15)
    y=heading('Ablação 1 - Coeficiente fixo',y)
    alpha=predictions['alpha_choice']
    y=para('<b>Previsão.</b> '+escape(predictions['fixed']),y)
    y=para(f'<b>Escolha e resultado.</b> Alpha fixo = {dec(alpha["value"],3)}; mediana observada de '
           f'{dec(alpha["tail_median"],4)} nos passos 80-100 mil das três seeds, faixa '
           f'{dec(alpha["tail_min"],4)}-{dec(alpha["tail_max"],4)}. '+prose['fixed_outcome']+
           f' O valor é selecionado em {ref("self.alpha = self.log_alpha.exp().item() if cfg.autotune")}; '
           f'com autotune desligado, o bloco de ajuste em {ref("if cfg.autotune:")} é pulado.',y)
    y=heading('Ablação 2 - Q1 no lugar de min(Q1, Q2)',y)
    y=para('<b>Previsão.</b> '+escape(predictions['single']),y)
    y=para('<b>Resultado e código.</b> '+prose['single_outcome']+
           f' A seleção em {ref("pick_q2 =")} envia Q1 para os dois argumentos do mínimo no alvo '
           f'({ref("next_soft_value =")}) e no ator ({ref("actor_loss = compute_actor_loss")}). '
           'Q2 ainda é calculado e treinado; esta ablação remove a escolha conservadora, sem eliminar a rede.',y)
    footer(2,y)

    y=header('Parte 5 - SAC x PPO com o orçamento solicitado')
    y=para('<b>Previsão anterior ao PPO.</b> '+escape(predictions['comparison']),y)
    y=image('parte5',y)
    y=para('<b>Figura 3.</b> Duas seeds de cada algoritmo. O PPO usa ppo_pendulum com os overrides exatos do enunciado. '
           'O laço fornecido completa rollouts de 4.096 e, por isso, realiza 98.304 dos 100.000 passos planejados; '
           'SAC realiza 100.000. SPS é acumulado; num_updates conta passos conjuntos dos críticos no SAC '
           'e minibatches conjuntos de ator/crítico no PPO.',y,small=True,gap=15)
    rows=[['<b>Run</b>','<b>Passos efetivos</b>','<b>Treino final</b>',
           '<b>Avaliação final*</b>','<b>Tempo (min)</b>','<b>SPS final</b>','<b>num_updates</b>']]
    compared=[f'{final}-s{s}' for s in (1,2)]+[f'ppo-hopper-s{s}' for s in (1,2)]
    for run in compared:
        r=records[run]
        m=r['final_metrics']
        rows.append([run,str(r['completed_steps']),dec(m[RET]),dec(m['eval/mean_return']),
                     dec(r['train_seconds']/60),str(m['charts/SPS']),str(m['charts/num_updates'])])
    y=table(rows,[WIDTH*v for v in (.22,.135,.135,.15,.12,.10,.14)],y)
    y=para('* SAC: ação média; PPO: avaliação amostrada padrão. A comparação principal usa as curvas de treino. '
           'O PPO também normaliza observações e recompensas, enquanto o SAC usa recompensas brutas.',y,small=True)
    common=min(records[r]['train_seconds'] for r in compared)
    values='; '.join(run+': '+dec(value_at(run,common)) for run in compared)
    y=para(f'<b>Mesmo tempo, {dec(common/60)} min:</b> valores do último log dentro do intervalo comum - '
           +escape(values)+'. Sem extrapolar curvas depois que um treino terminou.',y)
    y=para('<b>O que se confirmou.</b> '+prose['comparison_outcome'],y)
    footer(3,y)

    y=header('Mecanismos, reuso e reprodução dos resultados')
    y=heading('Quais peças explicam as diferenças observadas?',y)
    for key,label,pattern in [
        ('replay','Replay off-policy','idx = np.random.randint'),
        ('entropy','Entropia e temperatura','def compute_alpha_loss'),
        ('critics','Dois críticos','next_soft_value ='),
        ('actor','Ator reparametrizado','pre_tanh = distribution.rsample()')]:
        y=para(f'<b>{label}.</b> '+prose['mechanism'][key]+f' Fonte: {ref(pattern)}.',y)
    y=heading('Quantas vezes cada transição participa do gradiente?',y)
    reuse_rows=[['<b>Run / contador no último log</b>','<b>Amostras / transição armazenada ou coletada</b>']]
    for run in compared:
        r=records[run]
        cfg=r['config']
        m=r['final_metrics']
        if cfg['algorithm']=='sac':
            amount=m['charts/num_updates']*cfg['sac']['batch_size']/m['charts/buffer_size']
            detail=f'{dec(amount)} participações em minibatches de Q; aproximadamente o mesmo no ator.'
        else:
            amount=cfg['ppo_continuous_action']['update_epochs']
            detail=f'{amount} passagens por transição, em minibatches conjuntos do ator e crítico.'
        reuse_rows.append([run,detail])
    y=table(reuse_rows,[150,WIDTH-150],y,padding=6)
    y=para('SAC = num_updates × batch_size / buffer_size, no último log. A amostragem é com reposição; '
           'transições antigas entram mais vezes. policy_frequency=2 realiza dois passos do ator a cada dois de Q, '
           f'compensando o atraso ({ref("for _ in range(cfg.policy_frequency)")}). '
           'No candidato updates_per_step=2, o número de usos cresce aproximadamente duas vezes, com custo de relógio.',y)
    y=heading('Validação e limites da comparação',y)
    y=para('Somente os dois blocos pedidos foram preenchidos no SAC. A correção em train.py exibe policy_lr/q_lr, '
           'pois o SAC não tem learning_rate. Cada candidato altera uma chave; ablações partem do config selecionado. '
           f'CPU {escape(environment["cpu"])}, uma thread por run e até seis treinos simultâneos: '
           'tempos refletem essa concorrência. Três seeds na busca e duas nas ablações não demonstram superioridade geral.',y)
    y=para('Q menos retorno usa o retorno soft da política em evolução, não o Q ótimo. Diferenças também refletem '
           'trajetórias visitadas e não estacionariedade; leia-o junto com retorno e Q1.',y)
    y=heading('Código, configurações, previsões e dados',y)
    y=para(f'<b>Fork:</b> <link href="{publication["fork_url"]}" color="#247592">'
           'github.com/LeonardoLimaz/t-zero-learning</link>. Commit '+commit[:12]+'. '
           'Inclui código, configs final/candidatos/ablações/PPO, comandos em experimentos/plano.json '
           'e previsões em previsoes.json.',y)
    y=para(f'<b>Projeto W&amp;B público:</b> <link href="{publication["project_url"]}" color="#247592">'
           'sac-assignment - todos os 19 runs</link>. '
           f'<b>Relatório interativo:</b> <link href="{escape(publication["report_url"])}" color="#247592">'
           'curvas e resultados identificados</link>. Cada valor da tabela foi conferido com o run publicado.',y)
    y=para('<b>Referência de implementação:</b> '
           '<link href="https://docs.cleanrl.dev/rl-algorithms/sac/" color="#247592">'
           'CleanRL, Soft Actor-Critic</link>. As conclusões e gráficos deste relatório usam medições próprias.',y,small=True)
    footer(4,y)
    c.save()
    print(OUT/'relatorio_sac.pdf')


if __name__=='__main__':
    main()
