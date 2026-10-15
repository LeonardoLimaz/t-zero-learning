"""Verify project and run endpoints without loading credentials or using netrc."""
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
records=json.loads((ROOT/'output'/'resumo.json').read_text(encoding='utf-8'))
query='''query PublicAssignment($name: String!, $entity: String!) {
  project(name:$name,entityName:$entity) {
    name access runs(first:100) { edges { node { name displayName summaryMetrics } } }
  }
}'''
session=requests.Session()
session.trust_env=False
response=session.post('https://api.wandb.ai/graphql',json={'query':query,'variables':{
    'name':os.environ.get('WANDB_PROJECT','sac-assignment'),
    'entity':os.environ.get('WANDB_ENTITY','leonardo-leonardo2-federal-university-of-goi-s')}},timeout=60)
response.raise_for_status()
data=response.json()
if data.get('errors'):
    raise RuntimeError(data['errors'])
project=data['data']['project']
if project['access']!='USER_READ':
    raise RuntimeError(f'Unexpected visibility: {project["access"]}')
remote={e['node']['name']:e['node'] for e in project['runs']['edges']}
for local in records:
    run=remote[local['run_id']]
    if run['displayName']!=local['run']:
        raise RuntimeError('Display name mismatch')
    summary=json.loads(run['summaryMetrics'])
    for metric in ('eval/mean_return','charts/episodic_return_mean_last100'):
        if abs(summary[metric]-local['final_metrics'][metric])>1e-5:
            raise RuntimeError(f'Public endpoint mismatch {local["run"]}: {metric}')
print(f'Anonymous public access confirmed for {len(records)} runs, with matching final metrics.')
