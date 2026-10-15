"""Export original offline W&B histories; never interpolate the measured rows."""
import json
from pathlib import Path
import re

import numpy as np
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experimentos" / "resultados"


def read_history(path):
    store = DataStore()
    store.open_for_scan(str(path))
    rows = []
    try:
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("history"):
                rows.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                             for item in record.history.item})
    finally:
        store.close()
    return rows


def export(folder):
    metadata_path = folder / "command.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata['status'] != 'complete':
        return None
    files = list((folder / "wandb").glob("offline-run-*/*.wandb"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one W&B data file: {folder.name}")
    rows = read_history(files[0])
    (folder / "wandb_history.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True)+"\n" for row in rows), encoding="utf-8")
    config_paths = list((ROOT / "runs" / metadata['run']).glob("*/config.yml"))
    # A sandbox service failure can have written a config but no model/run.
    config_paths = [p for p in config_paths if (p.parent / "model.pt").exists()]
    if len(config_paths) != 1:
        raise RuntimeError(f"Expected one completed model config for {metadata['run']}")
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    metrics = {}
    for row in rows:
        metrics.update({k: v for k, v in row.items() if not k.startswith('_')})
    training = [r for r in rows if 'charts/SPS' in r]
    if 'eval/mean_return' not in metrics or not training:
        raise RuntimeError(f"Missing evaluation/training history: {folder.name}")
    log = (folder / "train.log").read_text(encoding="utf-8", errors="replace")
    evals = [float(x) for x in re.findall(r"eval_episode=\d+, episodic_return=([-+\d.eE]+)", log)]
    expected = 20 if config['algorithm'] == 'sac' else 10
    if len(evals) != expected:
        raise RuntimeError(f"Expected {expected} eval episodes, found {len(evals)} in {folder.name}")
    if abs(float(np.mean(evals[:10])) - metrics['eval/mean_return']) > 1e-5:
        raise RuntimeError(f"Evaluation log/history mismatch: {folder.name}")
    summary = dict(run=metadata['run'], phase=metadata['phase'], config=config,
                   offline_path=str(files[0].parent), run_id=files[0].stem.removeprefix('run-'),
                   duration_seconds=metadata['duration_seconds'], n_history=len(rows),
                   train_seconds=training[-1]['_runtime'],
                   completed_steps=max(int(r.get('_step', 0)) for r in rows),
                   final_metrics=metrics, eval_returns=evals,
                   run_dir=str(config_paths[0].parent))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summaries = []
    for folder in sorted(RESULTS.iterdir()):
        if folder.is_dir() and (result := export(folder)) is not None:
            summaries.append(result)
            m = result['final_metrics']
            print(f"{folder.name}: eval={m['eval/mean_return']:.1f}, "
                  f"train={m.get('charts/episodic_return_mean_last100', float('nan')):.1f}, "
                  f"alpha={m.get('losses/alpha', float('nan')):.4f}, "
                  f"steps={result['completed_steps']}, seconds={result['train_seconds']:.0f}")
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "resumo.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
