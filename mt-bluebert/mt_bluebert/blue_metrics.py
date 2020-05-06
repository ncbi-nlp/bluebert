# Copyright (c) Microsoft. All rights reserved.
# modified by: Yifan Peng
import logging
from enum import Enum

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

from mt_bluebert.data_utils.vocab import Vocabulary
from mt_bluebert.pmetrics import blue_classification_report, ner_report_conlleval


def compute_acc(predicts, labels):
    return accuracy_score(labels, predicts)


def compute_f1(predicts, labels):
    return f1_score(labels, predicts)


def compute_mcc(predicts, labels):
    return matthews_corrcoef(labels, predicts)


def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return pcof


def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return scof


def compute_auc(predicts, labels):
    auc = roc_auc_score(labels, predicts)
    return auc


def compute_micro_f1(predicts, labels):
    report = blue_classification_report(labels, predicts)
    return report.micro_row.f1.item()


def compute_micro_f1_subindex(predicts, labels, subindex):
    report = blue_classification_report(labels, predicts)
    try:
        sub_report = report.sub_report(subindex)
        return sub_report.micro_row.f1.item()
    except Exception as e:
        logging.error('%s\n%s', e, report.report())
        return 0

def compute_macro_f1_subindex(predicts, labels, subindex):
    report = blue_classification_report(labels, predicts)
    try:
        sub_report = report.sub_report(subindex)
        return sub_report.macro_row.f1.item()
    except Exception as e:
        logging.error('%s\n%s', e, report.report())
        return 0


def compute_seq_f1(predicts, labels, label_mapper):
    y_true, y_pred = [], []

    def trim(predict, label):
        temp_1 = []
        temp_2 = []

        # label_index = 1
        # pred_index = 1
        # while pred_index < len(predict) and label_index < len(label):
        #     if label_mapper[label[label_index]] == 'X' and label_mapper[predict[pred_index]] == 'X':
        #         label_index += 1
        #         pred_index += 1
        #     elif label_mapper[predict[pred_index]] == 'X':
        #         pred_index += 1
        #     elif label_mapper[label[label_index]] == 'X':
        #         label_index += 1
        #         pred_index += 1
        #     else:
        #         temp_1.append(label_mapper[label[label_index]])
        #         temp_2.append(label_mapper[predict[pred_index]])
        #         label_index += 1
        #         pred_index += 1

        for j, m in enumerate(predict):
            if j == 0:
                continue
            # if j >= len(label):
            #     print(predict, label)
            #     exit(1)

            if label_mapper[label[j]] != 'X':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)

    for i, (predict, label) in enumerate(zip(predicts, labels)):
        # try:
        trim(predict, label)
        # if i == 100:
        #     break
        # except Exception as e:
        #     print('index ', i)
        #     exit(1)
    report = ner_report_conlleval(y_true, y_pred)
    return report.micro_row.f1.item()


class BlueMetric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    MicroF1 = 8
    MicroF1WithoutLastOne = 9
    MacroF1WithoutLastOne = 10


METRIC_FUNC = {
    BlueMetric.ACC: compute_acc,
    BlueMetric.F1: compute_f1,
    BlueMetric.MCC: compute_mcc,
    BlueMetric.Pearson: compute_pearson,
    BlueMetric.Spearman: compute_spearman,
    BlueMetric.AUC: compute_auc,
    BlueMetric.SeqEval: compute_seq_f1,
    BlueMetric.MicroF1: compute_micro_f1,
    BlueMetric.MicroF1WithoutLastOne: compute_micro_f1_subindex,
    BlueMetric.MacroF1WithoutLastOne: compute_macro_f1_subindex
}


def calc_metrics(metric_meta, golds, predictions, scores, label_mapper: Vocabulary = None):
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (BlueMetric.ACC, BlueMetric.F1, BlueMetric.MCC, BlueMetric.MicroF1):
            metric = metric_func(predictions, golds)
        elif mm == BlueMetric.SeqEval:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == BlueMetric.MicroF1WithoutLastOne:
            metric = metric_func(predictions, golds, subindex=list(range(len(label_mapper) - 1)))
        elif mm == BlueMetric.MacroF1WithoutLastOne:
            metric = metric_func(predictions, golds, subindex=list(range(len(label_mapper) - 1)))
        else:
            if mm == BlueMetric.AUC:
                assert len(scores) == 2 * len(golds), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics
