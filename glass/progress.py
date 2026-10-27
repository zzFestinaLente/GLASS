"""Small atomic progress snapshots for independent GPU workers."""

import json
import os
from pathlib import Path
import time


PREFIX = 'GLASS_PROGRESS '


class StageProgress:
    """Track completed work through structured worker progress snapshots.

    Workers write beside their output, outside local-vector directories. A
    missing progress_path disables file output, including in direct API calls.
    """

    def __init__(self, config, stage, total, unit='items'):
        if total < 0:
            raise ValueError('Progress total must be nonnegative')
        self.path = Path(config['progress_path']) if config.get('progress_path') else None
        self.stage, self.total, self.unit = stage, total, unit
        self.done = 0
        self.status = 'loading'
        self.user = None
        self.extra = {}
        self.started = time.monotonic()
        self.closed = False

    def __enter__(self):
        self._write()
        return self

    def start(self):
        self.status = 'running'
        self._write()

    def update(self, n=1, user=None, **extra):
        if n < 0 or self.done + n > self.total:
            raise ValueError(f'Invalid {self.stage} progress: {self.done}+{n}/{self.total}')
        self.done += n
        self.status = 'running'
        if user is not None:
            self.user = str(user)
        self.extra.update(extra)
        self._write()

    def close(self, success=True):
        if not self.closed:
            incomplete = success and self.done != self.total
            if incomplete:
                success = False
            self.status = 'complete' if success else 'failed'
            self._write()
            self.closed = True
            if incomplete:
                raise RuntimeError(f'Incomplete {self.stage} progress: {self.done}/{self.total}')

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close(success=exc_type is None)
        except OSError:
            if exc_type is None:
                raise
        return False

    def _write(self):
        if self.path is None:
            return
        data = dict(stage=self.stage, done=self.done, total=self.total,
                    unit=self.unit, status=self.status, user=self.user,
                    elapsed_seconds=time.monotonic() - self.started, **self.extra)
        temporary = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        temporary.replace(self.path)


def worker_snapshot(stage, paths, exit_codes, previous=None):
    """Read all workers; retain the last good snapshot if a read races a write."""
    states = list(previous) if previous is not None else [None] * len(paths)
    for index, path in enumerate(paths):
        try:
            value = json.loads(Path(path).read_text(encoding='utf-8'))
            if (isinstance(value, dict) and value.get('stage') == stage
                    and isinstance(value.get('done'), int)
                    and isinstance(value.get('total'), int)
                    and not isinstance(value['done'], bool)
                    and not isinstance(value['total'], bool)
                    and value.get('status') in ('loading', 'running', 'complete', 'failed')
                    and isinstance(value.get('unit'), str) and value['unit']
                    and 0 <= value['done'] <= value['total']):
                states[index] = value
        except (OSError, ValueError):
            pass
    ready = sum(value is not None for value in states)
    total = sum(value['total'] for value in states) if ready == len(paths) else None
    done = sum(value['done'] for value in states if value is not None)
    if any(code not in (None, 0) for code in exit_codes):
        status = 'failed'
    elif any(value and value['status'] == 'failed' for value in states):
        status = 'failed'
    elif total is not None and done == total and all(value['status'] == 'complete' for value in states):
        status = 'finalizing'
    elif any(value and value['status'] == 'running' for value in states):
        status = 'running'
    else:
        status = 'loading'
    unit = next((value['unit'] for value in states if value is not None), 'items')
    return dict(stage=stage, done=done, total=total, unit=unit, status=status,
                workers_ready=ready, workers_done=sum(code == 0 for code in exit_codes),
                workers_total=len(paths)), states


def emit_progress(event):
    print(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
