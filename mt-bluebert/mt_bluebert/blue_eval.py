"""
Usage:
    blue_eval [options] --task_def=<file> --data_dir=<dir> --range=<str> --model_dir=<dir>

Options:
    --task_def=<file>
    --data_dir=<dir>
    --test_datasets=<str>
    --range=<str>
    --output=<file>
"""
import collections
import functools
import json
from pathlib import Path
from typing import Dict

import docopt as docopt
import numpy as np
import pandas as pd
import yaml

from mt_bluebert.blue_metrics import compute_micro_f1, compute_micro_f1_subindex, compute_seq_f1, compute_pearson


class TaskMetric:
    def __init__(self, task):
        self.task = task
        self.epochs = []
        self.scores = []

    def print_max(self):
        index = int(np.argmax(self.scores))
        print('%s: Max at epoch %d: %.3f' % (self.task, self.epochs[index], self.scores[index]))


def get_score1(pred_file, task, n_class, rows, golds, metric_func):
    with open(pred_file, 'r', encoding='utf8') as fp:
        obj = json.load(fp)

    preds = []
    for i, uid in enumerate(obj['uids']):
        if uid != rows[i]['uid']:
            raise ValueError('{}: {} vs {}'.format(task, uid, rows[i]['uid']))
        if n_class == 1:
            pred = obj['scores'][i]
        elif n_class > 1:
            pred = obj['predictions'][i]
        else:
            raise KeyError(task)
        preds.append(pred)

    score = metric_func(preds, golds)
    return score


def get_score2(pred_file, task, n_class, rows, golds, metric_func):
    with open(pred_file, 'r', encoding='utf8') as fp:
        objs = []
        for line in fp:
            objs.append(json.loads(line))

    preds = []
    for i, obj in enumerate(objs):
        if obj['uid'] != rows[i]['uid']:
            raise ValueError('{}: {} vs {}'.format(task, obj['uid'], rows[i]['uid']))
        if n_class == 1:
            pred = obj['score']
        elif n_class > 1:
            pred = obj['prediction']
        else:
            raise KeyError(task)
        preds.append(pred)

    score = metric_func(preds, golds)
    return score


def eval_blue(test_datasets, task_def_path, data_dir, model_dir, epochs):
    with open(task_def_path) as fp:
        task_def = yaml.safe_load(fp)

    METRIC_FUNC = {
        'biosses': compute_pearson,
        'clinicalsts': compute_pearson,
        'mednli': compute_micro_f1,
        'i2b2-2010-re': functools.partial(
            compute_micro_f1_subindex,
            subindex=[i for i in range(len(task_def['i2b2-2010-re']['labels']) - 1)]),
        'chemprot': functools.partial(
            compute_micro_f1_subindex,
            subindex=[i for i in range(len(task_def['chemprot']['labels']) - 1)]),
        'ddi2013-type': functools.partial(
            compute_micro_f1_subindex,
            subindex=[i for i in range(len(task_def['ddi2013-type']['labels']) - 1)]),
        'shareclefe': functools.partial(
            compute_seq_f1,
            label_mapper={i: v for i, v in enumerate(task_def['shareclefe']['labels'])}),
        'bc5cdr-disease': functools.partial(
            compute_seq_f1,
            label_mapper={i: v for i, v in enumerate(task_def['bc5cdr-disease']['labels'])}),
        'bc5cdr-chemical': functools.partial(
            compute_seq_f1,
            label_mapper={i: v for i, v in enumerate(task_def['bc5cdr-chemical']['labels'])}),
    }

    total_scores = collections.OrderedDict()  # type: Dict[str, TaskMetric]
    for task in test_datasets:
        n_class = task_def[task]['n_class']

        file = data_dir / f'{task}_test.json'
        with open(file, 'r', encoding='utf-8') as fp:
            rows = [json.loads(line) for line in fp]
            # print('Loaded {} samples'.format(len(rows)))
            golds = [row['label'] for row in rows]

        task_metric = TaskMetric(task)
        for epoch in epochs:
            # pred_file = model_dir / f'{task}_test_scores_{epoch}.json'
            # score = get_score1(pred_file, task, n_class, rows, golds, METRIC_FUNC[task])
            # scores.append(score)

            # if task in ('clinicalsts', 'i2b2-2010-re', 'mednli', 'shareclefe'):
            pred_file = model_dir / f'{task}_test_scores_{epoch}_2.json'
            try:
                score = get_score2(pred_file, task, n_class, rows, golds, METRIC_FUNC[task])
            except FileNotFoundError:
                pred_file = model_dir / f'{task}_test_scores_{epoch}.json'
                score = get_score1(pred_file, task, n_class, rows, golds, METRIC_FUNC[task])

            task_metric.epochs.append(epoch)
            task_metric.scores.append(score)

        total_scores[task] = task_metric
        task_metric.print_max()

        # if len(scores2) != 0:
        #     index = np.argmax(scores2)
        #     print('%s: Max at epoch %d: %.3f' % (task, epochs[index], scores2[index]))
        # index = np.argmin(scores)
        # print('%s: Min at epoch %d: %.3f' % (task, epochs[index], scores[index]))
    return total_scores


def pretty_print(total_scores, epochs, dest=None):
    # average
    for task_metric in total_scores.values():
        assert len(task_metric.scores) == len(epochs)
    avg_scores = [np.average([total_scores[t].scores[i] for t in total_scores.keys()])
                  for i, _ in enumerate(epochs)]
    index = int(np.argmax(avg_scores))
    print('On average, max at epoch %d: %.3f' % (epochs[index], avg_scores[index]))
    for t in total_scores:
        print('    %s: At epoch %d: %.3f' % (t, epochs[index], total_scores[t].scores[index]))

    if dest is not None:
        table = {'epoch': epochs, 'average': avg_scores}
        for t, v in total_scores.items():
            table[t] = v.scores
        df = pd.DataFrame.from_dict(table)
        df.to_csv(dest, index=None)


def main():
    args = docopt.docopt(__doc__)
    print(args)

    task_def_path = Path(args['--task_def'])
    model_dir = Path(args['--model_dir'])
    data_dir = Path(args['--data_dir'])

    toks = args['--range'].split(',')
    epochs = list(range(int(toks[0]), int(toks[1])))

    test_datasets = args['--test_datasets']
    if test_datasets is None:
        test_datasets = ['clinicalsts', 'i2b2-2010-re', 'mednli', 'chemprot', 'ddi2013-type',
                         'bc5cdr-chemical', 'bc5cdr-disease', 'shareclefe']
    else:
        test_datasets = test_datasets.split(',')

    total_scores = eval_blue(test_datasets, task_def_path, data_dir, model_dir, epochs)
    pretty_print(total_scores, epochs, args['--output'])


if __name__ == '__main__':
    main()
