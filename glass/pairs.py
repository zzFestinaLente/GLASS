"""Stage2: generate and filter positive/negative anchor contexts."""

import hashlib
import json
import random
from pathlib import Path

from rouge_score.rouge_scorer import RougeScorer

from .io import load_json, load_pickle, load_vectors, select_users, write_json
from .model import SteeringModel
from .progress import StageProgress
from .prompts import history_text, instruction, negative_history_text, normalize, wrap
from .rank import anchor_cluster_id, prime


def anchor_rng(seed, user_id, sample_id, attempt):
    """Independent random streams keep negatives stable across user sharding."""
    key = json.dumps([seed, str(user_id), str(sample_id), attempt], separators=(',', ':'))
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))


def negative_history(dataset, data, user_id, k, rng):
    others = [uid for uid, samples in data.items() if str(uid) != str(user_id) and samples]
    if not others:
        raise ValueError('Negative contexts require at least one other user with training samples')
    examples = [rng.choice(data[rng.choice(others)]) for _ in range(k)]
    return '\n'.join(negative_history_text(dataset, item, index)
                     for index, item in enumerate(examples))


def construct_user_pairs(config, user_id, samples, data, ranked):
    dataset = config['dataset']
    k = config.get('k', 5 if dataset.startswith('LaMP_') else 3)
    attempts = config.get('num_negatives', 2 if dataset.startswith('LaMP_') else 3)
    for index, sample in enumerate(samples):
        if config.get('limit_cases') is not None and index >= config['limit_cases']:
            break
        cluster_id = anchor_cluster_id(sample)
        positive_history = '\n'.join(history_text(dataset, item, i)
                                     for i, item in enumerate(ranked[str(user_id)][str(sample['id'])][:k]))
        for attempt in range(attempts):
            negative = negative_history(dataset, data, user_id, k,
                anchor_rng(config.get('seed', 42), user_id, sample['id'], attempt))
            pair = {'id': sample['id'], 'chosen': sample['output'],
                    'question_pos': instruction(dataset, sample['input'], positive_history),
                    'question_neg': instruction(dataset, sample['input'], negative),
                    'profile_cluster_id': cluster_id}
            yield pair


def generate_branch(model, dataset, questions, max_new_tokens):
    """Generate and normalize responses to the supplied questions."""
    messages = model.generate([wrap(dataset, text) for text in questions], max_new_tokens)
    if len(messages) != len(questions):
        raise RuntimeError('Generation returned a different number of samples')
    # LaMP responses are stripped before first-line selection.
    # LongLaMP responses go directly through first-line normalization.
    if dataset.startswith('LaMP_'):
        messages = [text.replace('<|eot_id|>', '').replace('<|end_of_text|>', '').strip()
                    for text in messages]
    return [normalize(text) for text in messages]


def fixed_input_rows(rows):
    """Reuse question inputs and training gold; regenerate answers and scores."""
    for row in rows:
        pair = {key: row[key] for key in ['id', 'question_pos', 'question_neg', 'chosen']}
        pair['profile_cluster_id'] = anchor_cluster_id(row)
        yield pair


def pair_count(config, samples, batch_size, fixed=False):
    """Count planned pairs without drawing negative contexts or formatting prompts."""
    limit = config.get('limit_cases')
    if fixed:
        if limit is not None:
            return len(samples[:((limit + batch_size - 1) // batch_size) * batch_size])
        return len(samples)
    attempts = config.get('num_negatives', 2 if config['dataset'].startswith('LaMP_') else 3)
    anchors = sum(limit is None or index < limit for index in range(len(samples)))
    return anchors * len(range(attempts))


def run(config, model=None):
    """Generate contrastive responses and write accepted/rejected pairs.

    Positive generation uses the user's global direction; negative generation
    uses the base model. Gain is positive ROUGE-L minus negative ROUGE-L.
    Pairs with gain at least min_gain are retained, including ties at zero.
    """
    positive_only = config.get('positive_only', False)
    output_path = Path(config['output_path'])
    error_path = Path(config['error_output_path']) if config.get('error_output_path') else None
    if not positive_only and error_path is None:
        raise ValueError('error_output_path is required for positive/negative filtering')
    if error_path is not None and output_path.resolve() == error_path.resolve():
        raise ValueError('Accepted and rejected outputs must be different paths')
    for path in [p for p in [output_path, error_path] if p is not None]:
        if path.exists():
            raise FileExistsError(path)
    dataset = config['dataset']
    fixed = config.get('fixed_pairs_path')
    if fixed:
        inputs = load_json(fixed)
        if not isinstance(inputs, dict):
            raise TypeError('fixed_pairs_path must contain a user-to-records dictionary')
        selected = select_users(inputs, config)
        data, ranked = None, None
    else:
        data = load_pickle(config['input_path'])
        ranked = load_json(config['ranked_path'])
        selected = select_users(data, config)
    for samples in selected.values():
        for sample in samples:
            anchor_cluster_id(sample)
    vectors = load_vectors(config['global_vectors_path'])
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    batch_size = int(config.get('batch_size', 1 if dataset.startswith('LaMP_') else 2))
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    total = sum(pair_count(config, samples, batch_size, bool(fixed))
                for samples in selected.values())
    with StageProgress(config, 'pairs', total, unit='pairs') as progress:
        model = model or SteeringModel(config['model_path'], config.get('device', 'cuda'))
        progress.start()
        accepted, rejected = {}, {}
        for user_id, samples in selected.items():
            accepted[user_id], rejected[user_id] = [], []
            records = list(fixed_input_rows(samples)) if fixed else list(
                construct_user_pairs(config, user_id, samples, data, ranked))
            if fixed and config.get('limit_cases') is not None:
                n = config['limit_cases']
                # Include the complete generation batch containing the last selected row.
                records = records[:((n + batch_size - 1) // batch_size) * batch_size]
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                prime(model, vectors, user_id, config)
                positives = generate_branch(model, dataset, [row['question_pos'] for row in batch],
                                            config['max_new_tokens'])
                model.reset()
                if positive_only:
                    for row, positive in zip(batch, positives):
                        accepted[user_id].append(dict(row, positive=positive,
                            pos_rouge=scorer.score(row['chosen'], positive)['rougeL'].fmeasure))
                    progress.update(len(batch), user=user_id)
                    continue
                negatives = generate_branch(model, dataset, [row['question_neg'] for row in batch],
                                            config['max_new_tokens'])
                for row, positive, negative in zip(batch, positives, negatives):
                    if dataset.startswith('LaMP_') and (not positive or not negative):
                        continue
                    result = dict(row, positive=positive, negative=negative,
                        pos_rouge=scorer.score(row['chosen'], positive)['rougeL'].fmeasure,
                        neg_rouge=scorer.score(row['chosen'], negative)['rougeL'].fmeasure)
                    gain = result['pos_rouge'] - result['neg_rouge']
                    reason = ('empty_negative' if not negative.strip() else
                              'empty_chosen' if not row['chosen'].strip() else None)
                    if reason is not None:
                        result['rejection_reason'] = reason
                    target = (accepted if reason is None and gain >= config.get('min_gain', 0.0)
                              else rejected)
                    target[user_id].append(result)
                progress.update(len(batch), user=user_id)
            if positive_only:
                print(f'Stage2 user {user_id}: {len(accepted[user_id])} positive generations; filtering skipped', flush=True)
            else:
                print(f'Stage2 user {user_id}: {len(accepted[user_id])} accepted, '
                      f'{len(rejected[user_id])} rejected', flush=True)
        model.reset()
        write_json(output_path, accepted)
        if error_path is not None:
            write_json(error_path, rejected)
    return {'accepted': accepted, 'rejected': rejected}
