import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from cgap_sbl_jadce.execution_control import TaskStopped, pause_point, update_status
from cgap_sbl_jadce.gui.jobs import config_digest


def run(folder):
    folder = Path(folder)

    for _ in range(100):
        record = json.loads((folder / 'job.json').read_text())
        if record.get('pid') == os.getpid():
            break
        time.sleep(.05)
    else:
        return 1

    def stop(signum, frame):
        raise TaskStopped()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        request = record['request']
        if config_digest(request['config'], request) != record['config_digest']:
            raise ValueError('Configuration changed before the worker started.')
        from cgap_sbl_jadce.gui.inputs import input_signature
        if request['kind'] == 'task':
            if record.get('complete'):
                from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
                plan = completion_plan(request['config'], request['task'], request.get('conditions'))
                if plan.blocked:
                    raise ValueError('Some saved conditions are incompatible. Select them and explicitly Rerun.')
            from cgap_sbl_jadce.recovery import begin_recovery
            from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
            reset = (request['task'] in ESTIMATORS and request.get('action', 'run') == 'run'
                     and not record.get('complete', False))
            begin_recovery(request, resume=record.get('resume', False), reset=reset)
        update_status(folder, status='running', input_signature=input_signature(request))
        pause_point()
        if request['kind'] == 'task':
            from cgap_sbl_jadce.tasks import run_task, task_actions
            task_id, action = request['task'], request['action']
            print(task_actions(task_id)[action].description.capitalize(), flush=True)
            from cgap_sbl_jadce.config import load_config
            cfg = load_config(request['config'])
            cfg['task_runtime'] = {'input_config': request['config'], 'action': action}
            paths = run_task(task_id, cfg, action=action, conditions=request.get('conditions'), methods=request.get('methods'),
                             realization_index=request.get('realization_index'), condition=request.get('condition'),
                             resume=record.get('resume', False) or record.get('complete', False))
        elif request['kind'] == 'preview':
            from cgap_sbl_jadce.gui.previews import render_preview
            paths = render_preview(request, folder / 'figures')
        else:
            raise ValueError('Unknown job type')
        if request['kind'] == 'task':
            from cgap_sbl_jadce.recovery import finish_recovery
            finish_recovery(request)
            if record.get('complete'):
                from cgap_sbl_jadce.gui.algorithm_actions import clear_finished_recoveries
                clear_finished_recoveries(request['config'], request['task'])
        update_status(folder, status='completed', paths=[str(Path(path).resolve()) for path in paths])
        print('Completed.', flush=True)
        return 0
    except TaskStopped:
        update_status(folder, status='stopped')
        print('Stopped. Use Resume to reuse compatible checkpoints.', flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        update_status(folder, status='failed', error=str(error))
        return 1


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
