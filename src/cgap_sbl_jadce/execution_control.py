"""Optional GUI controls checked between completed calculations, never inside a solver."""
import json
import os
from pathlib import Path
import time

from cgap_sbl_jadce.io import save_json


class TaskStopped(BaseException):
    """Leave the current worker without discarding committed checkpoints."""


def update_status(folder: Path, **changes):
    path = folder / 'job.json'
    record = json.loads(path.read_text())
    record.update(changes, updated=time.time())
    save_json(path, record)


def pause_point():
    location = os.environ.get('CGAP_JOB_DIRECTORY')
    if not location:
        return
    folder = Path(location)
    paused = False
    while True:
        command = json.loads((folder / 'control.json').read_text())['command']
        if command == 'stop':
            raise TaskStopped()
        if command != 'pause':
            if paused:
                update_status(folder, status='running')
            return
        if not paused:
            print('Paused at a completed calculation boundary.', flush=True)
            update_status(folder, status='paused')
            paused = True
        time.sleep(.2)
