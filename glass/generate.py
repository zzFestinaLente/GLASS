"""Stage3: combine the global prior and the selected local cluster direction."""

from pathlib import Path

from .io import load_pickle, load_vectors, select_users, user_vector, write_json
from .local_vectors import cluster_direction
from .model import SteeringModel
from .prompts import inference_prompt, normalize
from .progress import StageProgress


def strip_quotes(text):
    """Remove surrounding quotation marks after first-line selection."""
    quotes = set('"\'“”‘’„‟‚‛「」『』《》〈〉«»‹›〝〞＂＇')
    while text:
        old = text
        if text.startswith(('\\"', "\\'")):
            text = text[2:].lstrip()
        elif text[0] in quotes:
            text = text[1:].lstrip()
        if text.endswith(('\\"', "\\'")):
            text = text[:-2].rstrip()
        elif text and text[-1] in quotes:
            text = text[:-1].rstrip()
        if text == old:
            break
    return text


def run(config, model=None):
    """Generate predictions from assigned user-history clusters.

    Each question is decoded separately. Parallel execution assigns whole users
    to workers and retains this batch size.
    Questions with no local direction use the global prior.
    A zero strength disables its branch, including vector loading and injection.
    """
    output = Path(config['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    global_layer, local_layer = config.get('global_layer', 19), config.get('local_layer', 23)
    if global_layer >= local_layer:
        raise ValueError('The global layer must precede the local layer')
    data = select_users(load_pickle(config['input_path']), config)
    use_global = config['global_strength'] != 0
    use_local = config['local_strength'] != 0
    globals_ = load_vectors(config['global_vectors_path']) if use_global else None
    total = sum(len(samples[:config['limit_cases']]) if config.get('limit_cases') is not None
                else len(samples) for samples in data.values())
    with StageProgress(config, 'generate', total, unit='questions') as progress:
        model = model or SteeringModel(config['model_path'], config.get('device', 'cuda'))
        progress.start()
        records = []
        for user, samples in data.items():
            clusters = {}
            # Load the user's local bank when one is available.
            if use_local:
                cache_path = Path(config['local_vectors_path']) / f'{user}_{local_layer}.pt'
                if cache_path.exists():
                    clusters = load_vectors(cache_path)
                else:
                    print(f'Stage3 user {user}: no local bank, no local steering', flush=True)
            global_direction = user_vector(globals_, user) if use_global else None
            if config.get('limit_cases') is not None:
                samples = samples[:config['limit_cases']]
            for sample in samples:
                model.reset()
                if use_local:
                    cluster = clusters.get(str(sample['profile_cluster_id']), {})
                    directions = cluster.get(local_layer, [])
                    local_direction = cluster_direction(directions)
                    if local_direction is not None:
                        model.set_vector(local_layer, config['local_strength'] * local_direction)
                if use_global:
                    model.set_vector(global_layer, config['global_strength'] * global_direction)
                raw, = model.generate([inference_prompt(config['dataset'], sample['input'])],
                                      config['max_new_tokens'])
                # Strip whitespace before selecting the first line of each prediction.
                raw = raw.replace('<|eot_id|>', '').replace('<|end_of_text|>', '').strip()
                text = strip_quotes(normalize(raw))
                records.append({'id': str(sample['id']), 'generation': text,
                                'output': sample['output']})
                progress.update(user=user)
            print(f'Stage3 user {user}: {len(samples)} predictions', flush=True)
        model.reset()
        result = {'task': config['dataset'], 'golds': records}
        write_json(output, result)
    return result
