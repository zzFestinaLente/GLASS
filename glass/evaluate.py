"""ROUGE F1/precision/recall and optional NLTK METEOR, without network calls."""

import numpy as np

from rouge_score.rouge_scorer import RougeScorer

from .io import load_json, write_json


def score_records(records, meteor=False):
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    details = []
    meteor_scores = []
    if meteor:
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score
    for index, record in enumerate(records):
        prediction, label = record['generation'], record['output']
        scores = scorer.score(label, prediction)
        detail = {'id': index, 'prediction': prediction, 'label': label}
        for name, key in [('rouge1', 'rouge-1'), ('rougeL', 'rouge-L')]:
            value = scores[name]
            detail[key], detail[key + '_p'], detail[key + '_r'] = (
                value.fmeasure, value.precision, value.recall)
        details.append(detail)
        if meteor:
            meteor_scores.append(float(meteor_score([word_tokenize(label.strip())],
                                                   word_tokenize(prediction.strip())))
                                 if prediction.strip() and label.strip() else 0.0)
    keys = ['rouge-1', 'rouge-L', 'rouge-1_p', 'rouge-L_p', 'rouge-1_r', 'rouge-L_r']
    averages = {key: float(np.mean([row[key] for row in details])) if details else 0.0 for key in keys}
    if meteor:
        averages['METEOR'] = sum(meteor_scores) / max(len(meteor_scores), 1)
    return {'avg': averages, 'detail': details}


def run(config):
    data = load_json(config['input_path'])
    result = score_records(data['golds'], meteor=config.get('meteor', False))
    write_json(config['output_path'], result)
    return result
