"""Record relevant runtime versions and CPU model, without credentials/hostname."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]
cpu=platform.processor()
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
except (ImportError,OSError):
    pass
record=dict(python=platform.python_version(),platform=platform.platform(),cpu=cpu,
            logical_processors=os.cpu_count(),device='CPU',threads_per_run=1,max_concurrent_runs=6,
            packages={name:metadata.version(name) for name in
                      ('torch','numpy','gymnasium','mujoco','wandb','PyYAML','moviepy')})
(ROOT/'experimentos'/'ambiente.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
