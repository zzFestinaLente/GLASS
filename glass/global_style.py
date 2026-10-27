"""Construct each user's global style direction vG from historical responses."""

from pathlib import Path

import torch

from .io import load_pickle, select_users, write_vectors
from .prompts import inference_prompt
from .progress import StageProgress


def history_pair(dataset, item):
    """Read the task question and response from a user-profile entry."""
    if dataset == 'LaMP_4':
        return item['text'], item['title']
    if dataset == 'LaMP_5':
        return item['abstract'], item['title']
    if dataset == 'topic_writing_temporal':
        return item['summary'], item['content']
    if dataset == 'product_review_temporal':
        question = (
            f'Generate the review text written by a reviewer who has a given an overall rating of "{item["overall"]}" '
            f'for a product with description "{item["description"]}". '
            f'The summary of the review text is "{item["summary"]}".'
        )
        return question, item['reviewText']
    raise ValueError(f'Unsupported task: {dataset}')


def answer_tokens(tokenizer, question, answer, max_length):
    """Separate-tokenize Q and A and retain answer-token positions themselves."""
    prompt = tokenizer(question + ('\n' if answer else ''), add_special_tokens=False).input_ids
    response = tokenizer(answer, add_special_tokens=False).input_ids if answer else []
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    if len(response) > max_length - len(bos):
        response = response[:max_length - len(bos)]
        prompt = []
    elif len(bos) + len(prompt) + len(response) > max_length:
        keep = max_length - len(bos) - len(response)
        # Slicing with keep=0 retains the full prompt because Python treats -0 as 0.
        prompt = prompt[-keep:]
    start = len(bos) + len(prompt)
    return bos + prompt + response, start, start + len(response)


@torch.no_grad()
def construct_global(model, sae, dataset, samples, *,
                     layer=19, max_profile_samples=20, max_length=512):
    """Encode response tokens and average each response, then average responses equally.

    Decode the resulting mean SAE activation into the global style direction.
    """
    if max_profile_samples < 1 or max_length < 2:
        raise ValueError('Require max_profile_samples >= 1 and max_length >= 2')
    if not samples or not samples[0].get('profile'):
        raise ValueError('A user must have a nonempty profile in its first training sample')
    features = []
    for item in samples[0]['profile'][-max_profile_samples:]:
        question, answer = history_pair(dataset, item)
        question = inference_prompt(dataset, question)
        ids, start, end = answer_tokens(model.tokenizer, question, answer, max_length)
        if start == end:
            raise ValueError('A historical response has no answer tokens')
        tokens = torch.tensor([ids], dtype=torch.long, device=model.device)
        model.reset()
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
        hidden = model.activations(layer)[0, start:end, :]
        # Encode all answer tokens together as a [tokens, hidden_size] matrix.
        latent = sae.encode(hidden.reshape(-1, hidden.shape[-1]))
        features.append(latent.view(1, end - start, -1).mean(dim=1))
    mean_latent = torch.cat([feature.to(sae.device) for feature in features], dim=0).mean(dim=0, keepdim=True)
    return sae.decode(mean_latent).detach().cpu().clone()


def run(config: dict):
    """Read raw train.pkl and write a new UID -> BF16 [1,H] global.pt file."""
    from .model import SteeringModel
    from .sae import OfficialSAE

    output = Path(config['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Choose a new output path: {output}')
    dataset = config['dataset']
    inference_prompt(dataset, '')  # Validate before loading model weights.
    users = select_users(load_pickle(config['input_path']), config)
    with StageProgress(config, 'global', len(users), unit='users') as progress:
        model = SteeringModel(config['model_path'], device=config.get('device', 'cuda'))
        sae = OfficialSAE(config['sae_model_path'], device=config.get('device', 'cuda'))
        progress.start()
        vectors = {}
        for user, samples in users.items():
            vectors[user] = construct_global(
                model, sae, dataset, samples,
                layer=config.get('global_layer', 19),
                max_profile_samples=config.get('max_profile_samples', 20),
                max_length=config.get('max_length', 512),
            )
            progress.update(user=user)
            print(f'global: {user} ({len(vectors)}/{len(users)})', flush=True)
        write_vectors(output, vectors)
    return vectors
