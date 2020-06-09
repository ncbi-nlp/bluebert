"""
Copyright (c) 2018, Yifan Peng
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import List

import numpy as np
from sklearn import metrics
from tabulate import tabulate
from mt_bluebert import conlleval


def _divide(x, y):
    try:
        return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float), where=y != 0)
    except:
        return np.nan


def tp_tn_fp_fn(confusion_matrix):
    fp = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    fn = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = np.sum(confusion_matrix) - (fp + fn + tp)
    return tp, tn, fp, fn


class PReportRow:
    """https://en.wikipedia.org/wiki/Precision_and_recall"""

    def __init__(self, category, **kwargs):
        self.category = category
        self.tp = kwargs.pop('tp', np.nan)
        self.tn = kwargs.pop('tn', np.nan)
        self.fp = kwargs.pop('fp', np.nan)
        self.fn = kwargs.pop('fn', np.nan)
        self.precision = kwargs.pop('precision', _divide(self.tp, self.tp + self.fp))
        self.recall = kwargs.pop('recall', _divide(self.tp, self.tp + self.fn))
        self.f1 = kwargs.pop('f1', _divide(2 * self.precision * self.recall, self.precision + self.recall))
        self.specificity = kwargs.pop('specificity', _divide(self.tn, self.tn + self.fp))
        self.support = kwargs.pop('support', self.tp + self.fn)
        self.accuracy = kwargs.pop('accuracy', _divide(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn))
        self.balanced_accuracy = kwargs.pop('balanced_accuracy', _divide(self.recall + self.specificity, 2))
        # different names
        self.sensitivity = self.recall
        self.positive_predictive_value = self.precision
        self.true_positive_rate = self.recall
        self.true_negative_rate = self.specificity


class PReport:
    def __init__(self, rows: List[PReportRow]):
        self.rows = rows
        self.micro_row = self.compute_micro()
        self.macro_row = self.compute_macro()

    def compute_micro(self):
        tps = [row.tp for row in self.rows]
        tns = [row.tn for row in self.rows]
        fps = [row.fp for row in self.rows]
        fns = [row.fn for row in self.rows]
        return PReportRow('micro', tp=np.sum(tps), tn=np.sum(tns), fp=np.sum(fps), fn=np.sum(fns))

    def compute_macro(self):
        ps = [row.precision for row in self.rows]
        rs = [row.recall for row in self.rows]
        fs = [row.f1 for row in self.rows]
        return PReportRow('macro', precision=np.average(ps), recall=np.average(rs), f1=np.average(fs))

    def report(self, digits=3, micro=False, macro=False):
        headers = ['Class', 'TP', 'FP', 'FN',
                   'Precision', 'Recall', 'F-score',
                   'Support']
        float_formatter = ['g'] * 4 + ['.{}f'.format(digits)] * 3 + ['g']

        rows = self.rows
        if micro:
            rows.append(self.micro_row)
        if macro:
            rows.append(self.macro_row)

        table = [[r.category, r.tp, r.fp, r.fn, r.precision, r.recall, r.f1, r.support] for r in rows]
        return tabulate(table, showindex=False, headers=headers,
                        tablefmt="plain", floatfmt=float_formatter)

    def sub_report(self, subindex) -> 'PReport':
        rows = [self.rows[i] for i in subindex]
        return PReport(rows)


def ner_report_conlleval(y_true: List[List[str]], y_pred: List[List[str]]) -> PReport:
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
    """
    lines = []
    assert len(y_true) == len(y_pred)
    for t_sen, p_sen in zip(y_true, y_pred):
        assert len(t_sen) == len(p_sen)
        for t_word, p_word in zip(t_sen, p_sen):
            lines.append(f'XXX\t{t_word}\t{p_word}\n')
        lines.append('\n')

    counts = conlleval.evaluate(lines)
    overall, by_type = conlleval.metrics(counts)

    rows = []
    for i, m in sorted(by_type.items()):
        rows.append(PReportRow(i, tp=m.tp, fp=m.fp, fn=m.fn))
    return PReport(rows)


def blue_classification_report(y_true, y_pred, *_, **kwargs) -> PReport:
    """
    Args:
        y_true: (n_sample, )
        y_pred: (n_sample, )
    """
    classes_ = kwargs.get('classes_', None)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    tp, tn, fp, fn = tp_tn_fp_fn(confusion_matrix)

    if classes_ is None:
        classes_ = [i for i in range(confusion_matrix.shape[0])]

    rows = []
    for i, c in enumerate(classes_):
        rows.append(PReportRow(c, tp=tp[i], tn=tn[i], fp=fp[i], fn=fn[i]))

    report = PReport(rows)
    return report

