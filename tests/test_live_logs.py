import io

from tqdm import tqdm

from cgap_sbl_jadce.gui.jobs import Jobs


def log_file(tmp_path):
    jobs = Jobs(tmp_path)
    folder = jobs.folder / 'sample'
    folder.mkdir()
    return jobs, folder / 'worker.log'


def test_nested_bars_strip_cursor_codes_and_preserve_attached_messages(tmp_path):
    jobs, path = log_file(tmp_path)
    path.write_text('Run CoSaMP\n\rMMV-CoSaMP conditions:   0%|          | 0/11 [00:00<?, ?condition/s]\n'
                    '\r\x1b[32mMMV-CoSaMP L=30 SNR=4 dB:  50%|█████     | 503/1000 [02:50<02:56, 2.81realization/s]'
                    '\x1b[0m\x1b[APaused at a completed calculation boundary.\n')
    text, progress = jobs.log('sample')
    assert text == 'Run CoSaMP\nPaused at a completed calculation boundary.'
    assert progress == {'label': 'MMV-CoSaMP L=30 SNR=4 dB', 'done': 503, 'total': 1000}


def test_partial_progress_writes_do_not_leak_into_live_text(tmp_path):
    jobs, path = log_file(tmp_path)
    path.write_text('Starting\n\rSBL tuning:  50%|#####     | 1/2 [00:01<00:01, 1setup/s]'
                    '\rSBL tuning: 100%|####\x1b[')
    text, progress = jobs.log('sample')
    assert text == 'Starting'
    assert progress == {'label': 'SBL tuning', 'done': 1, 'total': 2}
    with path.open('a') as stream:
        stream.write('A######| 2/2 [00:02<00:00, 1setup/s]\nCompleted.\n')
    text, progress = jobs.log('sample')
    assert text == 'Starting\nCompleted.'
    assert progress['done'] == progress['total'] == 2


def test_bounded_tail_discards_first_fragment_and_keeps_errors(tmp_path):
    jobs, path = log_file(tmp_path)
    path.write_text('Old data\n' + '█' * 22000 + '| 9/100 [00:00<?, ?it/s]\x1b[A\n'
                    '\rSBL:  10%|█         | 10/100 [00:01<00:09, 10it/s]'
                    'RuntimeWarning: test warning\nTraceback (most recent call last):\n'
                    'ValueError: Threshold 25% exceeds limit\n')
    text, progress = jobs.log('sample')
    assert text == ('RuntimeWarning: test warning\nTraceback (most recent call last):\n'
                    'ValueError: Threshold 25% exceeds limit')
    assert progress == {'label': 'SBL', 'done': 10, 'total': 100}
    assert '�' not in text and '\x1b' not in text


def test_actual_nested_tqdm_output_is_rendered_as_progress_only(tmp_path):
    jobs, path = log_file(tmp_path)
    stream = io.StringIO()
    stream.write('Starting\n')
    for _ in tqdm(range(2), desc='SBL conditions', file=stream, mininterval=0):
        for _ in tqdm(range(3), desc='SBL realizations', file=stream, mininterval=0, leave=False):
            pass
    stream.write('Completed.\n')
    path.write_text(stream.getvalue())
    text, progress = jobs.log('sample')
    assert text == 'Starting\nCompleted.'
    assert progress == {'label': 'SBL conditions', 'done': 2, 'total': 2}


def test_regular_messages_are_preserved_and_display_is_bounded(tmp_path):
    jobs, path = log_file(tmp_path)
    assert jobs.log('missing') == ('', None)
    path.write_text('\n'.join(f'Calculation {i}' for i in range(120)))
    text, progress = jobs.log('sample')
    assert text.splitlines() == [f'Calculation {i}' for i in range(20, 120)]
    assert progress is None
