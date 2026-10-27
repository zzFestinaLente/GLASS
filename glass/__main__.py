"""One entry point with plain JSON task/stage settings."""

import argparse
import importlib
import json

from .io import load_json
from .paths import expand_setting, load_paths


STAGES = ('prepare', 'cluster', 'global', 'rank', 'pairs', 'local', 'generate', 'evaluate')


def load_config(path, stage, overrides=()):
    document = load_json(path)
    config = {**document.get('common', {}), **document[stage]}
    overridden = set()
    for item in overrides:
        key, separator, value = item.partition('=')
        if not separator:
            raise ValueError(f'Expected key=value: {item}')
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        config[key] = value
        overridden.add(key)
    paths = load_paths()
    return {key: expand_setting(value, paths)
            if not document.get('paths_resolved', False) or key in overridden else value
            for key, value in config.items()}


def run_stage(stage, config):
    if stage in ('prepare', 'cluster'):
        from . import data
        return getattr(data, 'run_' + stage)(config)
    module = {'global': 'global_style', 'local': 'local_style'}.get(stage, stage)
    return importlib.import_module('.' + module, __package__).run(config)


def main():
    parser = argparse.ArgumentParser(description='GLASS: profile preparation, style vectors, and generation')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', required=True, help='Task JSON with common and stage sections')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a setting; values may be JSON literals')
    parser.add_argument('--gpus', nargs='+', help='Run whole-user shards on these visible GPU numbers or UUIDs')
    args = parser.parse_args()
    config = load_config(args.config, args.stage, args.set)
    if args.gpus:
        from .parallel import run_parallel
        run_parallel(args.stage, config, args.gpus)
    else:
        run_stage(args.stage, config)


if __name__ == '__main__':
    main()
