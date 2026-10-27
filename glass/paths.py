"""Persistent local paths and project-relative data/output defaults."""

import os
from pathlib import Path
import re

from .io import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = {
    'model_path': 'models/Meta-Llama-3-8B',
    'sae_model_path': 'models/sae-llama-3-8b-32x-v2/layers.19',
    'data_dir': 'artifacts',
    'run_dir': 'runs',
}
PATH_VARIABLES = {
    'GLASS_MODEL': 'model_path',
    'GLASS_SAE': 'sae_model_path',
    'GLASS_DATA': 'data_dir',
    'GLASS_RUN': 'run_dir',
}
VARIABLE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)')


def load_paths(project_root=None):
    """Read configs/paths.json without creating any directories.

    Relative file settings are anchored to the source root. Explicit environment
    settings take precedence, so launchers can pass their resolved paths to workers.
    """
    root = Path(project_root).expanduser().resolve() if project_root is not None else PROJECT_ROOT
    path = root / 'configs' / 'paths.json'
    settings = load_json(path) if path.is_file() else {}
    if not isinstance(settings, dict) or settings.keys() - DEFAULT_PATHS.keys():
        raise ValueError(f'{path} must be an object with keys from {list(DEFAULT_PATHS)}')
    result = {}
    for variable, key in PATH_VARIABLES.items():
        value = os.environ.get(variable, settings.get(key, DEFAULT_PATHS[key]))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Expected a nonempty path for {key} in {path}')
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        result[variable] = str(resolved.resolve())
    return result


def expand_setting(value, paths):
    """Expand placeholders once; absolute paths are already resolved values."""
    if not isinstance(value, str):
        return value
    if Path(value).is_absolute():
        return value
    variables = {**os.environ, **paths}

    def replace(match):
        name = match.group(1) or match.group(2)
        if name not in variables:
            raise ValueError(f'Undefined path variable {name}: {value}')
        return variables[name]

    return os.path.expanduser(VARIABLE.sub(replace, value))
