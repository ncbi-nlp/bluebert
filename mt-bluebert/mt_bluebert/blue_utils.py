import csv
import json
import logging

from mt_bluebert.data_utils import DataFormat


def load_relation(file):
    rows = []
    with open(file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, blocks in enumerate(reader):
            assert len(blocks) == 3, '%s:%s: number of blocks: %s' % (file, i, len(blocks))
            lab = blocks[-1]
            sample = {'uid': blocks[0], 'premise': blocks[1], 'label': lab}
            rows.append(sample)
    return rows


def load_mednli(file):
    """MEDNLI for classification"""
    rows = []
    with open(file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, blocks in enumerate(reader):
            assert len(blocks) == 4, '%s:%s: number of blocks: %s' % (file, i, len(blocks))
            lab = blocks[0]
            assert lab is not None, '%s:%s: label is None' % (file, i)
            sample = {'uid': blocks[1], 'premise': blocks[2], 'hypothesis': blocks[3], 'label': lab}
            rows.append(sample)
    return rows


def load_sts(file):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, blocks in enumerate(reader):
            assert len(blocks) > 8, '%s:%s: number of blocks: %s' % (file, i, len(blocks))
            score = blocks[-1]
            sample = {'uid': cnt, 'premise': blocks[-3],'hypothesis': blocks[-2], 'label': score}
            rows.append(sample)
            cnt += 1
    return rows


def load_ner(file,  sep='\t'):
    rows = []
    sentence = []
    label = []
    offset = []
    uid = None
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "\n":
                if len(sentence) > 0:
                    assert uid is not None
                    sample = {'uid': uid, 'premise': sentence, 'label': label, 'offset': offset}
                    rows.append(sample)
                    sentence = []
                    label = []
                    offset = []
                    uid = None
                continue
            splits = line.split(sep)
            assert len(splits) == 4
            sentence.append(splits[0])
            offset.append('{};{}'.format(int(splits[2]), int(splits[2]) + len(splits[0])))
            label.append(splits[3])
            if splits[1] != '-':
                uid = splits[1] + '.' + splits[2]
        if len(sentence) > 0:
            assert uid is not None
            sample = {'uid': uid, 'premise': sentence, 'label': label, 'offset': offset}
            rows.append(sample)
    return rows


def dump_PremiseOnly(rows, out_path):
    logger = logging.getLogger(__name__)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            row_str = []
            for col in ["uid", "label", "premise"]:
                if "\t" in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s: %s has tab' % (out_path, i, col))
                row_str.append(str(row[col]))
            out_f.write('\t'.join(row_str) + '\n')


def dump_PremiseAndOneHypothesis(rows, out_path):
    logger = logging.getLogger(__name__)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            row_str = []
            for col in ["uid", "label", "premise", "hypothesis"]:
                if "\t" in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s: %s has tab' % (out_path, i, col))
                row_str.append(str(row[col]))
            out_f.write('\t'.join(row_str) + '\n')


def dump_Sequence(rows, out_path):
    logger = logging.getLogger(__name__)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            row_str = []
            if "\t" in str(row['uid']):
                row['uid'] = row['uid'].replace('\t', ' ')
                logger.warning('%s:%s: %s has tab' % (out_path, i, 'uid'))
            row_str.append(str(row['uid']))
            for col in ["label", "premise", "offset"]:
                for j, token in enumerate(row[col]):
                    if "\t" in str(token):
                        row[col][j] = token.replace('\t', ' ')
                        logger.warning('%s:%s: %s has tab' % (out_path, i, col))
                row_str.append(json.dumps(row[col]))
            out_f.write('\t'.join(row_str) + '\n')


def dump_PremiseAndMultiHypothesis(rows, out_path):
    logger = logging.getLogger(__name__)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            row_str = []
            for col in ["uid", "label", "premise"]:
                if "\t" in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s: %s has tab' % (out_path, i, col))
                row_str.append(str(row[col]))
            hypothesis = row["hypothesis"]
            for j, one_hypo in enumerate(hypothesis):
                if "\t" in str(one_hypo):
                    hypothesis[j] = one_hypo.replace('\t', ' ')
                    logger.warning('%s:%s: hypothesis has tab' % (out_path, i))
            row_str.append("\t".join(hypothesis))
            out_f.write('\t'.join(row_str) + '\n')

def dump_rows(rows, out_path, data_format: DataFormat):
    """
    output files should have following format
    """
    if data_format == DataFormat.PremiseOnly:
        dump_PremiseOnly(rows, out_path)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        dump_PremiseAndOneHypothesis(rows, out_path)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        dump_PremiseAndMultiHypothesis(rows, out_path)
    elif data_format == DataFormat.Sequence:
        dump_Sequence(rows, out_path)
    else:
        raise ValueError(data_format)

