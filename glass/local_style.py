"""Construct ordered local contrastive vectors vC, grouped by history cluster."""

from pathlib import Path

import torch

from .io import load_json, load_vectors, select_users, user_vector, write_vectors
from .prompts import extraction_prompt
from .progress import StageProgress


def clean_pair_question(question):
    """Remove generation-only instruction suffixes before local extraction."""
    for marker in (
        'No explanation needed;',
        "Now it's your turn:No explanation and additional words needed; just give your answer directly.",
    ):
        if marker in question:
            question = question.split(marker, 1)[0].rstrip()
    return question


def response_tokens(model, dataset, question, answer):
    if not isinstance(answer, str):
        raise TypeError('Contrastive responses must be strings')
    prompt = extraction_prompt(dataset, clean_pair_question(question))
    # Tokenize the prompt and answer jointly without truncation. The answer
    # boundary is the length of the separately tokenized prompt.
    start = model.tokenizer.encode(prompt, return_tensors='pt').shape[1]
    tokens = model.tokenizer.encode(prompt + answer, return_tensors='pt')
    return tokens, start


@torch.no_grad()
def response_mean(model, dataset, question, answer, layer, inputs=None):
    tokens, start = inputs if inputs is not None else response_tokens(model, dataset, question, answer)
    model.model(tokens.to(model.device))
    return model.activations(layer)[0, start:, :].mean(0).detach().cpu()


@torch.no_grad()
def construct_local(model, dataset, pairs, global_vector, *, global_layer=19,
                    local_layer=23, priming_strength=0.1, progress=None):
    """Positive (gold chosen, vG primed) minus negative (generated, unprimed)."""
    if global_layer >= local_layer:
        raise ValueError('The global priming layer must precede the local extraction layer')
    clusters = {}
    for pair in pairs:
        positive_inputs = response_tokens(model, dataset, pair['question_pos'], pair['chosen'])
        negative_inputs = response_tokens(model, dataset, pair['question_neg'], pair['negative'])
        if any(tokens.shape[1] <= start for tokens, start in (positive_inputs, negative_inputs)):
            model.reset()
            print(f"local: skipped pair {pair['id']} cluster {pair['profile_cluster_id']}: "
                  'empty answer-token span', flush=True)
            if progress is not None:
                progress.update()
            continue
        model.reset()
        model.set_vector(global_layer, priming_strength * global_vector)
        positive = response_mean(model, dataset, pair['question_pos'], pair['chosen'], local_layer, positive_inputs)
        model.reset()
        negative = response_mean(model, dataset, pair['question_neg'], pair['negative'], local_layer, negative_inputs)
        # Subtraction happens after the two BF16 response means move to CPU.
        direction = positive - negative
        if torch.isfinite(direction).all().item():
            clusters.setdefault(str(pair['profile_cluster_id']), {}).setdefault(local_layer, []).append(direction)
        else:
            print(f"local: skipped pair {pair['id']} cluster {pair['profile_cluster_id']}: "
                  'nonfinite direction', flush=True)
        if progress is not None:
            progress.update()
    model.reset()
    return clusters


def run(config: dict):
    """Read accepted pairs and global vectors; write one local-vector cache per user."""
    from .model import SteeringModel

    output = Path(config['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {output}')
    users = select_users(load_json(config['input_path']), config)
    global_vectors = load_vectors(config['global_vectors_path'])
    # Resolve global directions for the selected users.
    vectors = {user: user_vector(global_vectors, user) for user in users}
    total = sum(len(pairs) for pairs in users.values())
    with StageProgress(config, 'local', total, unit='pairs') as progress:
        model = SteeringModel(config['model_path'], device=config.get('device', 'cuda'))
        progress.start()
        output.mkdir(parents=True, exist_ok=False)
        paths = {}
        for user, pairs in users.items():
            progress.update(0, user=user)
            layer = config.get('local_layer', 23)
            clusters = construct_local(
                model, config['dataset'], pairs, vectors[user],
                global_layer=config.get('global_layer', 19), local_layer=layer,
                priming_strength=config['priming_strength'], progress=progress,
            )
            path = output / f'{user}_{layer}.pt'
            write_vectors(path, clusters)
            paths[user] = str(path)
            print(f'local: {user} ({len(paths)}/{len(users)})', flush=True)
    return paths
