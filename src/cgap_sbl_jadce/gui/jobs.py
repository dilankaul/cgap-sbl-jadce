"""Independent paused workers and one computing worker per workspace, with durable status and append-only logs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4

from cgap_sbl_jadce.io import save_json

ACTIVE = {'queued', 'running', 'paused'}
BUSY = {'queued', 'running'}
RECOVERABLE = {'failed', 'stopped', 'interrupted'}


def config_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_token(pid):
    """Linux process start time guards against PID reuse and zombie workers."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


class Jobs:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.folder = self.project / '.gui'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.children = {}

    @contextmanager
    def locked(self):
        with (self.folder / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def records(self):
        records = []
        for path in sorted(self.folder.glob('*/job.json'), reverse=True):
            record = json.loads(path.read_text())
            child = self.children.get(record['id'])
            if child is not None:
                child.poll()  # Reap completed children while this server is alive.
            if record['status'] in ACTIVE:
                token = process_token(record.get('pid'))
                if not token or token != record.get('process_token'):
                    # A queued record gets a short launch grace period.
                    if record['status'] != 'queued' or time.time() - record['created'] > 10:
                        record.update(status='interrupted', updated=time.time())
                        save_json(path, record)
            records.append(record)
        return sorted(records, key=lambda item: item['created'], reverse=True)

    def busy(self):
        return next((job for job in self.records() if job['status'] in BUSY or
                     (job['status'] == 'paused' and json.loads((self.folder / job['id'] / 'control.json').read_text())['command'] == 'run')), None)

    def active(self):
        return self.busy() or next((job for job in self.records() if job['status'] == 'paused'), None)

    def task_job(self, config, task, action='run'):
        config = str(Path(config).resolve())
        return next((job for job in self.records() if job['request'].get('kind') == 'task'
                     and job['request']['config'] == config and job['request']['task'] == task
                     and job['request'].get('action', 'run') == action), None)

    def start(self, request):
        """Requests identify existing tasks or saved-data previews, never shell code."""
        request = dict(request)
        config = Path(request['config'])
        config = (self.project / config).resolve() if not config.is_absolute() else config.resolve()
        if not config.is_file():
            raise ValueError(f'Configuration does not exist: {config}')
        request['config'] = str(config)
        with self.locked():
            if self.busy():
                raise ValueError('A worker is already active. Wait until its pause is confirmed or stop it first.')
            if request.get('kind') == 'task':
                same = self.task_job(config, request['task'], request.get('action', 'run'))
                if same and same['status'] in ACTIVE:
                    raise ValueError('This task already has paused work. Continue or stop that task before starting it again.')
            job_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
            folder = self.folder / job_id
            folder.mkdir()
            record = {'id': job_id, 'request': request, 'config_digest': config_digest(config),
                      'status': 'queued', 'created': time.time(), 'updated': time.time()}
            save_json(folder / 'job.json', record)
            save_json(folder / 'control.json', {'command': 'run'})
            env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', CGAP_JOB_DIRECTORY=str(folder))
            source = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [source, env.get('PYTHONPATH', '')]))
            try:
                with (folder / 'worker.log').open('ab') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'cgap_sbl_jadce.gui.worker', str(folder)],
                                             cwd=self.project, env=env, stdout=log, stderr=log,
                                             start_new_session=True)
                self.children[job_id] = child
                record.update(pid=child.pid, process_token=process_token(child.pid))
                save_json(folder / 'job.json', record)
            except OSError as error:
                record.update(status='failed', error=str(error))
                save_json(folder / 'job.json', record)
                raise
            return record

    def control(self, job_id, command):
        if command not in {'pause', 'run', 'stop'}:
            raise ValueError('Unknown worker control')
        with self.locked():
            record = next(job for job in self.records() if job['id'] == job_id)
            if record['status'] not in ACTIVE:
                raise ValueError('This worker is no longer running; use Restart to continue its checkpoints.')
            if command == 'run':
                busy = self.busy()
                if busy and busy['id'] != job_id:
                    raise ValueError('Pause the running task and wait for confirmation before continuing this task.')
                if config_digest(record['request']['config']) != record['config_digest']:
                    raise ValueError('Configuration changed while the worker was running. Stop it, then start with the new settings.')
                from cgap_sbl_jadce.gui.inputs import input_signature
                if record.get('input_signature') and input_signature(record['request']) != record['input_signature']:
                    raise ValueError('Required inputs changed while paused. Stop and restart this task to use compatible checkpoints.')
            save_json(self.folder / job_id / 'control.json', {'command': command})
            # Pausing is cooperative. Stopping interrupts the current unit, which
            # will be repeated on restart; previously committed units survive.
            if command == 'stop' and record.get('process_token') == process_token(record.get('pid')):
                try:
                    os.kill(record['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def restart(self, job_id):
        record = next(job for job in self.records() if job['id'] == job_id)
        if record['status'] not in RECOVERABLE:
            raise ValueError('Only interrupted, stopped, or failed workers can restart.')
        if config_digest(record['request']['config']) != record['config_digest']:
            raise ValueError('Configuration changed. Start a new task; only compatible checkpoints will be reused.')
        return self.start(record['request'])

    def log(self, job_id):
        path = self.folder / job_id / 'worker.log'
        if not path.exists():
            return '', None
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 64000))
            lines = stream.read().decode('utf-8', errors='replace').replace('\r', '\n').splitlines()
        progress = None
        visible = []
        for line in lines:
            line = re.sub(r'Task[ -]?0?\d+\s*', '', line)
            match = re.search(r'(.+?):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[', line)
            if match:
                progress = {'label': match[1], 'done': int(match[3]), 'total': int(match[4])}
            elif line.strip():
                visible.append(line)
        return '\n'.join(visible[-100:]), progress
