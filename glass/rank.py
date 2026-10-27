"""Stage1: score each same-user demonstration with the primed model."""

from copy import deepcopy
from pathlib import Path

from rouge_score.rouge_scorer import RougeScorer

from .io import load_pickle, load_vectors, select_users, user_vector, write_json
from .model import SteeringModel
from .progress import StageProgress
from .prompts import history_text, instruction, normalize, wrap


def prime(model, vectors, user_id, config):
    model.reset()
    vector = user_vector(vectors, user_id).to(model.device)
    model.set_vector(config.get('global_layer', 19), config['global_strength'] * vector)


def profile_entries(sample, user_samples):
    if sample.get('profile'):
        return sample['profile']
    return [item for item in user_samples if item is not sample]


def anchor_cluster_id(sample):
    """Read the target anchor's cluster ID."""
    if sample.get('profile_cluster_id') is None:
        raise ValueError(f'Training anchor {sample.get("id")!r} has no profile_cluster_id; '
                         'rebuild anchors with the cluster stage')
    return sample['profile_cluster_id']


def batches(samples, dataset, batch_size, limit_cases=None, limit_history=None):
    """Yield same-user generation batches containing selected target items."""
    batch = []
    for case_index, sample in enumerate(samples):
        cluster_id = anchor_cluster_id(sample)
        for history_index, history in enumerate(profile_entries(sample, samples)):
            text = history_text(dataset, history)
            target = ((limit_cases is None or case_index < limit_cases)
                      and (limit_history is None or history_index < limit_history))
            batch.append({'sample_id': str(sample['id']), 'history': history,
                          'profile_cluster_id': cluster_id,
                          'history_text': text, 'gold': sample['output'],
                          'prompt': wrap(dataset, instruction(dataset, sample['input'], text, stage='stage1')),
                          'target': target})
            if len(batch) == batch_size:
                if any(item['target'] for item in batch):
                    yield batch
                batch = []
    if batch and any(item['target'] for item in batch):
        yield batch


def generation_count(samples, batch_size, limit_cases=None, limit_history=None):
    """Count the selected complete batches without constructing their prompts."""
    total, pending, selected = 0, 0, False
    for case_index, sample in enumerate(samples):
        for history_index in range(len(profile_entries(sample, samples))):
            pending += 1
            selected = selected or (
                (limit_cases is None or case_index < limit_cases)
                and (limit_history is None or history_index < limit_history))
            if pending == batch_size:
                if selected:
                    total += pending
                pending, selected = 0, False
    return total + (pending if selected else 0)


def run(config, model=None):
    """Write rankings as ``user_id -> sample_id -> ranked histories``."""
    output_path = Path(config['output_path'])
    if output_path.exists():
        raise FileExistsError(output_path)
    dataset = config['dataset']
    data = select_users(load_pickle(config['input_path']), config)
    for user_id, samples in data.items():
        sample_ids = set()
        for sample in samples:
            anchor_cluster_id(sample)
            sample_id = str(sample['id'])
            if sample_id in sample_ids:
                raise ValueError(f'Duplicate training anchor ID {sample_id!r} for user {user_id!r}')
            sample_ids.add(sample_id)
    vectors = load_vectors(config['global_vectors_path'])
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    batch_size = int(config.get('batch_size', 2))
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    total = sum(generation_count(samples, batch_size, config.get('limit_cases'),
                                 config.get('limit_history')) for samples in data.values())
    with StageProgress(config, 'rank', total, unit='generations') as progress:
        model = model or SteeringModel(config['model_path'], config.get('device', 'cuda'))
        progress.start()
        ranked = {}
        for user_id, samples in data.items():
            user_ranked = ranked.setdefault(str(user_id), {})
            generated = 0
            for batch in batches(samples, dataset, batch_size,
                                 config.get('limit_cases'), config.get('limit_history')):
                prime(model, vectors, user_id, config)
                messages = model.generate([item['prompt'] for item in batch], config['max_new_tokens'])
                if len(messages) != len(batch):
                    raise RuntimeError('Generation returned a different number of samples')
                for item, message in zip(batch, messages):
                    generation = normalize(message)
                    record = deepcopy(item['history'])
                    record.update(profile_cluster_id=item['profile_cluster_id'],
                                  generation=generation, prompt_history=item['history_text'],
                                  rouge_l=scorer.score(item['gold'], generation)['rougeL'].fmeasure)
                    user_ranked.setdefault(item['sample_id'], []).append(record)
                generated += len(batch)
                progress.update(len(batch), user=user_id)
            print(f'Stage1 user {user_id}: {generated} generations', flush=True)
        for user_ranked in ranked.values():
            for histories in user_ranked.values():
                histories.sort(key=lambda item: item['rouge_l'], reverse=True)
        model.reset()
        write_json(output_path, ranked)
    return ranked
