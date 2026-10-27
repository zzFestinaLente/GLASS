"""Exact Base-Llama templates used by the four GLASS tasks."""

TASKS = ('LaMP_4', 'LaMP_5', 'product_review_temporal', 'topic_writing_temporal')
ANSWER_LABELS = {'LaMP_4': 'Headline', 'LaMP_5': 'Title',
                 'product_review_temporal': 'Review', 'topic_writing_temporal': 'Post'}
DIRECT = 'No explanation and additional words needed; just give your answer directly.'


def review_question(item):
    if 'input' in item:
        return item['input']
    return (
        f'Generate the review text written by a reviewer who has a given an overall rating of "{item["overall"]}" '
        f'for a product with description "{item["description"]}". '
        f'The summary of the review text is "{item["summary"]}".'
    )


def history_text(dataset, item, index=None):
    """None gives the one-shot Stage1 block; an index gives an n-shot block."""
    if dataset == 'LaMP_4':
        question, answer = item['text'], item['title']
    elif dataset == 'LaMP_5':
        question, answer = item['abstract'], item['title']
    elif dataset == 'product_review_temporal':
        question, answer = review_question(item), item['reviewText']
    elif dataset == 'topic_writing_temporal':
        question, answer = item['summary'], item['content']
    else:
        raise ValueError(f'Unsupported task: {dataset}')
    label = 'Historical sample' if index is None else f'Historical sample {index}'
    return f'{label}:\n Q: {question}. \n A: {answer}.'


def negative_history_text(dataset, item, index):
    """Other-user training anchors use input/output instead of profile fields."""
    question = item['input']
    if dataset == 'LaMP_4':
        question = question.replace('Generate a headline for the following article:', '', 1).lstrip()
    elif dataset == 'LaMP_5':
        question = question.replace('Generate a title for the following abstract of a paper:', '', 1).lstrip()
        question = question.replace('Generate a headline for the following article:', '', 1).lstrip()
    elif dataset not in TASKS:
        raise ValueError(f'Unsupported task: {dataset}')
    return f'Historical sample {index}:\n Q: {question}. \n A: {item["output"]}.'


def instruction(dataset, question, history=None, stage='stage2'):
    """Build the task instruction before the ### wrapper.

    Stage1/2 demonstrations use the same turn suffix for every task, with no
    newline between that suffix and DIRECT. Prompts without history omit it.
    """
    if dataset not in TASKS:
        raise ValueError(f'Unsupported task: {dataset}')
    question = question.rstrip()
    if dataset == 'LaMP_5' and question.startswith('Generate a headline for the following article:'):
        question = question.replace('Generate a headline for the following article:',
                                    'Generate a title for the following abstract of a paper:', 1)
    suffix = '' if question.endswith('.') else '.'
    if dataset == 'LaMP_4' and not question.startswith('Generate'):
        question = f'Generate a headline for the following article: {question}'
    elif dataset == 'LaMP_5' and not question.startswith('Generate'):
        question = f'Generate a title for the following abstract of a paper: {question}'
    elif dataset == 'topic_writing_temporal' and not question.startswith('Generate'):
        question = f'Generate the content for a reddit post {question}'
    text = question + suffix
    if history is not None:
        text += f"\nFor your reference, here are the user's past QA pairs:\n {history}\n"
        text += "Now it's your turn:"
    return text + DIRECT


def wrap(dataset, text):
    return f'### Instruction:\n{text}\n\n### {ANSWER_LABELS[dataset]}:\n'


def inference_prompt(dataset, question):
    return wrap(dataset, instruction(dataset, question))


def extraction_prompt(dataset, question):
    """Append the direct-answer instruction and wrap a cleaned pair prompt."""
    return wrap(dataset, question + DIRECT)


def normalize(text, first_line=True):
    if not text:
        return ''
    if first_line:
        text = text.split('\n', 1)[0].strip()
    text = text.replace('<|eot_id|>', '').replace('<|end_of_text|>', '').strip()
    return text.strip()
