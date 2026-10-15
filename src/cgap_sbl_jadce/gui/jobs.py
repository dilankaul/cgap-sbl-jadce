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

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_PROGRESS_FRAME = re.compile(
    r'([^\r\n]+?):\s*\d+(?:\.\d+)?%\|[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[[^\]\r\n]*\]')
_PARTIAL_PROGRESS = re.compile(r'^.*?:\s*\d+(?:\.\d+)?%\|')


def config_digest(path, request=None):
    request = request or {}
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.settings import request_settings
    if request.get('kind') not in {'task', 'preview'}:
        return hashlib.sha256(json.dumps(load_config(path), sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(json.dumps(request_settings(load_config(path), request), sort_keys=True).encode()).hexdigest()


def process_token(pid):
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
        self.saved_jobs = {}

    @contextmanager
    def locked(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def records(self):
        for child in self.children.values():
            child.poll()
        records = []
        for path in sorted(self.folder.glob('*/job.json'), reverse=True):
            record = json.loads(path.read_text())
            if record['status'] in ACTIVE:
                token = process_token(record.get('pid'))
                if not token or token != record.get('process_token'):

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

    def task_job(self, config, task, action='run', conditions=None):
        config = str(Path(config).resolve())
        from cgap_sbl_jadce.recovery import load_recoveries
        local = [job for job in self.records() if job['request'].get('kind') == 'task'
                 and job['request']['config'] == config and job['request']['task'] == task
                 and job['request'].get('action', 'run') == action]
        if conditions is not None:
            pairs = {tuple(pair) for pair in conditions}
            local = [job for job in local if {tuple(pair) for pair in job['request'].get('conditions', [])} == pairs]
        candidates = list(local)
        for saved in load_recoveries(config, task, action):
            if conditions is not None and {tuple(pair) for pair in saved['request'].get('conditions', [])} != pairs:
                continue
            if any(job['id'] == saved.get('job_id') or (job.get('resume') and job['request'] == saved['request']) for job in local):
                continue
            job_id = 'saved-' + hashlib.sha256(str(saved['path']).encode()).hexdigest()[:16]
            record = {'id': job_id, 'status': 'interrupted', 'portable': True,
                      'request': saved['request'], 'config_digest': saved['config_digest'],
                      'created': saved['created']}
            self.saved_jobs[job_id] = record
            candidates.append(record)

        candidates.sort(key=lambda job: job['created'], reverse=True)
        return next((job for job in candidates if job['status'] in ACTIVE), None) or (candidates[0] if candidates else None)

    def start(self, request, resume=False, complete=False):
        request = dict(request)
        config = Path(request['config'])
        config = (self.project / config).resolve() if not config.is_absolute() else config.resolve()
        if not config.is_file():
            raise ValueError(f'Configuration does not exist: {config}')
        request['config'] = str(config)
        if complete:
            self._validate_completion(request)
        if resume and request.get('kind') == 'task':
            from cgap_sbl_jadce.recovery import validate_recovery
            validate_recovery(request)
        with self.locked():
            if self.busy():
                raise ValueError('A worker is already active. Wait until its pause is confirmed or stop it first.')
            if request.get('kind') == 'task':
                for same in self.records():
                    previous = same['request']
                    if (same['status'] not in ACTIVE or previous.get('kind') != 'task' or
                            previous['config'] != str(config) or previous['task'] != request['task'] or
                            previous.get('action', 'run') != request.get('action', 'run')):
                        continue
                    old, new = previous.get('conditions'), request.get('conditions')
                    if old is None or new is None or {tuple(pair) for pair in old} & {tuple(pair) for pair in new}:
                        raise ValueError('This task already has paused work for these conditions. Resume or stop it first.')
            job_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
            folder = self.folder / job_id
            folder.mkdir()
            record = {'id': job_id, 'request': request, 'resume': resume, 'complete': complete, 'config_digest': config_digest(config, request),
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

    def _validate_completion(self, request):
        from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
        from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
        if request.get('kind') != 'task' or request['task'] not in ESTIMATORS or request.get('action', 'run') != 'run':
            raise ValueError('Completion applies only to algorithm evaluation.')
        plan = completion_plan(request['config'], request['task'], request.get('conditions'))
        if plan.blocked:
            names = ', '.join(f'L={L}, SNR={snr:g} dB' for L, snr in plan.blocked)
            raise ValueError(f'Incompatible saved work at {names}. Select those conditions and explicitly Rerun.')
        return plan

    def complete(self, request):
        request = dict(request, config=str(Path(request['config']).resolve()))
        self._validate_completion(request)
        if self.busy():
            raise ValueError('Pause the running task and wait for confirmation before completing these conditions.')
        from cgap_sbl_jadce.config import load_config
        cfg = load_config(request['config'])
        from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
        pairs = {value[:2] for value in evaluation_conditions(cfg, request.get('conditions'), method=request['task'])}


        paused = []
        for job in self.records():
            old = job['request']
            if (job['status'] != 'paused' or old.get('kind') != 'task' or
                    old['config'] != request['config'] or old['task'] != request['task'] or
                    old.get('action', 'run') != 'run'):
                continue
            if pairs & {value[:2] for value in evaluation_conditions(cfg, old.get('conditions'), method=old['task'])}:
                paused.append(job)
        for job in paused:
            self.control(job['id'], 'stop')
        deadline = time.monotonic() + 5
        while paused and time.monotonic() < deadline:
            alive = {job['id'] for job in self.records() if job['status'] in ACTIVE}
            if not any(job['id'] in alive for job in paused):
                break
            time.sleep(.05)
        else:
            if paused:
                raise ValueError('Paused workers are still stopping. Try Complete again once they have stopped.')
        return self.start(request, complete=True)

    def control(self, job_id, command):
        if command not in {'pause', 'run', 'stop'}:
            raise ValueError('Unknown worker control')
        with self.locked():
            record = next(job for job in self.records() if job['id'] == job_id)
            if record['status'] not in ACTIVE:
                raise ValueError('This worker is no longer running; use Resume to recover from its checkpoints.')
            if command == 'run':
                busy = self.busy()
                if busy and busy['id'] != job_id:
                    raise ValueError('Pause the running task and wait for confirmation before resuming this task.')
                if config_digest(record['request']['config'], record['request']) != record['config_digest']:
                    raise ValueError('Configuration changed while the worker was running. Stop it, then start with the new settings.')
                from cgap_sbl_jadce.gui.inputs import input_signature
                if record.get('input_signature') and input_signature(record['request']) != record['input_signature']:
                    raise ValueError('Required inputs changed while paused. Stop this task, then Resume to use compatible checkpoints.')
            save_json(self.folder / job_id / 'control.json', {'command': command})


            if command == 'stop' and record.get('process_token') == process_token(record.get('pid')):
                try:
                    os.kill(record['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def resume(self, job_id):
        record = self.saved_jobs.get(job_id) or next(job for job in self.records() if job['id'] == job_id)
        if record['status'] == 'paused':
            self.control(job_id, 'run')
            return record
        if record['status'] not in RECOVERABLE:
            raise ValueError('Only paused, interrupted, stopped, or failed tasks can resume.')
        if config_digest(record['request']['config'], record['request']) != record['config_digest']:
            raise ValueError('Configuration changed. Start a new task; only compatible checkpoints will be reused.')
        return self.start(record['request'], resume=True, complete=record.get('complete', False))

    def log(self, job_id):
        if job_id in self.saved_jobs:
            return 'Unfinished work found in saved recovery data. Resume uses the original condition selection.', None
        path = self.folder / job_id / 'worker.log'
        if not path.exists():
            return '', None
        with path.open('rb') as stream:
            offset = max(0, path.stat().st_size - 64000)
            stream.seek(offset)
            text = stream.read().decode('utf-8', errors='replace')
        if offset:
            boundary = re.search(r'[\r\n]', text)
            text = text[boundary.end():] if boundary else ''
        text = _ANSI_ESCAPE.sub('', text)
        text = re.sub(r'\x1b(?:\[[0-?]*[ -/]*)?$', '', text)
        progress = None
        visible = []
        for line in text.replace('\r', '\n').splitlines():
            match = _PROGRESS_FRAME.match(line)
            if match:
                progress = {'label': match[1].strip(), 'done': int(match[2]), 'total': int(match[3])}
                line = line[:match.start()] + line[match.end():]
            elif _PARTIAL_PROGRESS.match(line):
                continue
            if line.strip():
                visible.append(line.rstrip())
        return '\n'.join(visible[-100:]), progress
