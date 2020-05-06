from typing import Tuple

from data_utils.vocab import Vocabulary
from mt_bluebert.blue_metrics import calc_metrics, BlueMetric


def eval_model(model, data,
               metric_meta: Tuple[BlueMetric],
               use_cuda: bool=True,
               with_label: bool=True,
               label_mapper: Vocabulary=None):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids
