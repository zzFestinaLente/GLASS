"""Independent model processes over whole users; concatenate in input order."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .io import load_json, load_vectors, write_json, write_vectors
from .progress import emit_progress, worker_snapshot


def merge_dicts(parts):
    result = {}
    for part in parts:
        overlap = result.keys() & part.keys()
        if overlap:
            raise ValueError(f'Shards have overlapping keys: {list(overlap)[:3]}')
        result.update(part)
    return result


def run_parallel(stage, config, gpus):
    """Partition users among independent model workers."""
    if stage not in ('global', 'rank', 'pairs', 'local', 'generate'):
        raise ValueError('GPU sharding is supported for global/rank/pairs/local/generate')
    if not gpus:
        raise ValueError('Specify at least one GPU')
    if len(set(gpus)) != len(gpus):
        raise ValueError('Specify each GPU once')
    if config.get('num_shards', 1) != 1 or config.get('shard_index', 0) != 0:
        raise ValueError('Do not combine --gpus with manual shard settings')
    output = Path(config['output_path'])
    outputs = [output]
    errors = config.get('error_output_path')
    if errors and stage == 'pairs':
        outputs.append(Path(errors))
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    work = output.parent / (output.name + '.shards')
    work.mkdir(parents=True, exist_ok=False)
    workers = []
    failures = []
    try:
        # Save absolute paths so child configs describe exactly what was run.
        for index, gpu in enumerate(gpus):
            part = work / str(index)
            part.mkdir()
            child = dict(config, num_shards=len(gpus), shard_index=index, device='cuda:0',
                         output_path=str((part / ('vectors' if stage == 'local' else output.name)).resolve()),
                         progress_path=str((part / 'progress.json').resolve()))
            if errors and stage == 'pairs':
                child['error_output_path'] = str((part / 'rejected.json').resolve())
            child_path = part / 'config.json'
            write_json(child_path, {'paths_resolved': True, stage: child})
            log = (part / 'run.log').open('x')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            try:
                process = subprocess.Popen([sys.executable, '-m', 'glass', stage, '--config', str(child_path.resolve())],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append((process, log, child))
        states = None
        last_report = 0
        while True:
            exit_codes = [process.poll() for process, _, _ in workers]
            finished = all(code is not None for code in exit_codes)
            now = time.monotonic()
            if finished or now - last_report >= 1:
                event, states = worker_snapshot(stage, [child['progress_path'] for _, _, child in workers],
                                                exit_codes, states)
                emit_progress(event)
                last_report = now
            if finished:
                break
            time.sleep(0.25)
        for process, log, child in workers:
            code = process.wait()
            log.close()
            if code:
                failures.append((process.pid, code))
    except BaseException:
        for process, log, _ in workers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
        raise
    if failures:
        raise RuntimeError(f'Shard workers failed: {failures}; see {work}')
    emit_progress(dict(event, status='merging'))
    paths = [Path(child['output_path']) for _, _, child in workers]
    if stage == 'local':
        output.mkdir(parents=True, exist_ok=False)
        for path in paths:
            for source in path.iterdir():
                if source.suffix != '.pt':
                    raise ValueError(f'Unexpected local output: {source.name}')
                target = output / source.name
                if target.exists():
                    raise ValueError(f'Duplicate local cache: {target.name}')
                with source.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    elif stage == 'global':
        write_vectors(output, merge_dicts([load_vectors(path) for path in paths]))
    elif stage == 'generate':
        parts = [load_json(path) for path in paths]
        if any(part['task'] != config['dataset'] for part in parts):
            raise ValueError('Shard task mismatch')
        write_json(output, {'task': config['dataset'], 'golds': [row for part in parts for row in part['golds']]})
    else:
        write_json(output, merge_dicts([load_json(path) for path in paths]))
        if errors and stage == 'pairs':
            write_json(errors, merge_dicts([load_json(child['error_output_path']) for _, _, child in workers]))
    emit_progress(dict(event, status='complete'))
