"""Execute original training CLI with per-run logs and a bounded worker count."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experimentos" / "resultados"
START_LOCK = threading.Lock()


def run_job(job):
    ident = f"{job['case']}-s{job['seed']}"
    folder = RESULTS / ident
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "command.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if previous['status'] == 'complete':
            print(f"SKIP {ident}: complete", flush=True)
            return ident
        raise RuntimeError(f"Incomplete prior attempt for {ident}; inspect before rerunning")
    overrides = [f"exp_name={ident}", f"seed={job['seed']}", *job.get('overrides', [])]
    command = [sys.executable, "train.py", "--config", job['config'], "--override", *overrides]
    env = os.environ.copy()
    env.update({"WANDB_MODE": "offline", "WANDB_PROJECT": os.environ.get("WANDB_PROJECT","sac-assignment"),
                "WANDB_ENTITY": os.environ.get("WANDB_ENTITY","leonardo-leonardo2-federal-university-of-goi-s"),
                "WANDB_DIR": str(folder), "WANDB_SILENT": "true",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "TMP": str(folder), "TEMP": str(folder), "TMPDIR": str(folder)})
    started = time.time()
    algorithm=yaml.safe_load((ROOT/'configs'/f'{job["config"]}.yml').read_text(encoding='utf-8'))['algorithm']
    record = dict(job, run=ident, command=command, status="running",
                  queued_utc=datetime.now(timezone.utc).isoformat(),
                  algorithm_source=f'algorithms/{algorithm}.py',
                  algorithm_sha256=hashlib.sha256((ROOT / "algorithms" / f"{algorithm}.py").read_bytes()).hexdigest())
    while True:
        with START_LOCK:
            active = 0
            for path in RESULTS.glob('*/command.json'):
                try:
                    active += json.loads(path.read_text(encoding='utf-8')).get('status') == 'running'
                except json.JSONDecodeError:
                    active += 1
            if active < 6:
                record['started_utc']=datetime.now(timezone.utc).isoformat()
                metadata.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                break
        time.sleep(1)
    started = time.time()
    print(f"START {ident}", flush=True)
    with (folder / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(status="complete" if process.returncode == 0 else "failed",
                  exit_code=process.returncode, duration_seconds=round(time.time()-started, 2),
                  ended_utc=datetime.now(timezone.utc).isoformat())
    metadata.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"END {ident}: {record['status']} ({record['duration_seconds']} s)", flush=True)
    if process.returncode:
        print((folder / "train.log").read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)
        raise RuntimeError(f"{ident} exited with {process.returncode}")
    return ident


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / '.env')
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    jobs = json.loads((ROOT / "experimentos" / "plano.json").read_text(encoding="utf-8"))
    jobs = [job for job in jobs if job['phase'] == args.phase]
    if not jobs:
        raise ValueError(f"No jobs for phase {args.phase}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_job, job) for job in jobs]):
            future.result()


if __name__ == "__main__":
    main()
