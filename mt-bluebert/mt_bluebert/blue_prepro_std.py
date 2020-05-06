"""
Preprocessing BLUE dataset.

Usage:
    blue_prepro_std [options] --vocab=<file> --root_dir=<dir> --task_def=<file> --datasets=<str>

Options:
    --do_lower_case
    --max_seq_len=<int>     [default: 128]
    --overwrite
"""

# Copyright (c) Microsoft. All rights reserved.
# Modified by Yifan Peng

import json
import logging
import os

import docopt
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mt_bluebert.data_utils.log_wrapper import create_logger
from mt_bluebert.data_utils.task_def import TaskType, DataFormat, EncoderModelType
from mt_bluebert.data_utils.vocab import Vocabulary
from mt_bluebert.experiments.squad import squad_utils
from mt_bluebert.blue_exp_def import BlueTaskDefs
from mt_bluebert.mt_dnn.batcher import BatchGen

MAX_SEQ_LEN = 512


def load_data(file_path, data_format: DataFormat, task_type: TaskType, label_dict: Vocabulary = None):
    """
    Args:
        label_dict: map string label to numbers. only valid for Classification task or ranking task.
            For ranking task, better label should have large number
    """
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip().split("\t")
        if data_format == DataFormat.PremiseOnly:
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2]}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            assert len(fields) == 4
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2], "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        elif data_format == DataFormat.Sequence:
            assert len(fields) == 4
            row = {"uid": fields[0], "label": json.loads(fields[1]), "premise": json.loads(fields[2]),
                   "offset": json.loads(fields[3])}
        else:
            raise ValueError(data_format)

        if task_type == TaskType.Classification:
            if label_dict is not None:
                row["label"] = label_dict[row["label"]]
            else:
                row["label"] = int(row["label"])
        elif task_type == TaskType.Regression:
            row["label"] = float(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels
        elif task_type == TaskType.Span:
            pass  # don't process row label
        elif task_type == TaskType.SequenceLabeling:
            if label_dict is not None:
                row["label"] = [label_dict[l] for l in row["label"]]
            else:
                row["label"] = [int(l) for l in row["label"]]

        assert row["label"] is not None, \
            '%s: %s: label is None. label_dict: %s' % (file_path, row['uid'], label_dict.tok2ind)
        rows.append(row)
    return rows


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    logger = logging.getLogger(__name__)
    truncate_tokens_a = False
    truncate_tokens_b = False
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            truncate_tokens_a = True
            tokens_a.pop()
        else:
            truncate_tokens_b = True
            tokens_b.pop()

    if truncate_tokens_a:
        logger.debug('%s: longer than %s', tokens_a, max_length)
    if truncate_tokens_b:
        logger.debug('%s: longer than %s', tokens_b, max_length)


def bert_feature_extractor(text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
    logger = logging.getLogger(__name__)
    tokens_a = tokenize_fn.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenize_fn.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            logger.debug('%s: longer than %s', text_a, max_seq_length)
            tokens_a = tokens_a[:max_seq_length - 2]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    if tokens_b:
        input_ids = tokenize_fn.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'])
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    if tokens_b:
        input_ids = tokenize_fn.convert_tokens_to_ids(['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * (len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
    else:
        input_ids = tokenize_fn.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
    input_mask = None
    return input_ids, input_mask, segment_ids


def split_if_longer(data, label_mapper, max_seq_len=30):
    rows = []
    for idx, sample in enumerate(data):
        # uid = sample['uid']
        docid = sample['uid'].split('.')[0]
        premise = sample['premise']
        label = sample['label']
        offset = sample['offset']

        while len(premise) > max_seq_len:
            tmplabel = label[:max_seq_len]
            for iidx in range(len(tmplabel)):
                if label_mapper[tmplabel.pop()] == 'O':
                    break
            p = premise[:len(tmplabel) + 1]
            l = label[:len(tmplabel) + 1]
            o = offset[:len(tmplabel) + 1]
            uid = '{}.{}'.format(docid, o[0].split(';')[0])
            rows.append({"uid": uid, "label": l, "premise": p, "offset": o})

            premise = premise[len(tmplabel) + 1:]
            label = label[len(tmplabel) + 1:]
            offset = offset[len(tmplabel) + 1:]

        if len(premise) == 0:
            continue

        uid = '{}.{}'.format(docid, offset[0].split(';')[0])
        rows.append({"uid": uid, "label": label, "premise": premise, "offset": offset})
    return rows


def build_data_sequence(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, label_mapper=None):
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            ne_labels = sample['label']
            premise_offset = sample['offset']

            tokens = []
            labels = []
            offsets = []
            for i, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                tokens.extend(subwords)
                for j in range(len(subwords)):
                    if j == 0:
                        labels.append(ne_labels[i])
                        offsets.append(premise_offset[i])
                    else:
                        labels.append(label_mapper['X'])
                        offsets.append('X')

            if len(premise) > max_seq_len - 2:
                tokens = tokens[:max_seq_len - 2]
                labels = labels[:max_seq_len - 2]
                offsets = offsets[:max_seq_len - 2]

            label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
            offsets = ['X'] + offsets + ['X']
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            assert len(label) == len(input_ids)
            type_ids = [0] * len(input_ids)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'offset': offsets}
            writer.write('{}\n'.format(json.dumps(features)))


def build_data_premise_only(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None,
                            encoderModelType=EncoderModelType.BERT):
    """Build data of single sentence tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            label = sample['label']
            assert encoderModelType == EncoderModelType.BERT
            input_ids, _, type_ids = bert_feature_extractor(premise, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))


def build_data_premise_and_one_hypo(
        data, dump_path, task_type, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
    """Build data of sentence pair tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['label']
            assert encoderModelType == EncoderModelType.BERT
            input_ids, _, type_ids = bert_feature_extractor(
                premise, hypothesis, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
            if task_type == TaskType.Span:
                seg_a_start = len(type_ids) - sum(type_ids)
                seg_a_end = len(type_ids)
                answer_start, answer_end, answer, is_impossible = squad_utils.parse_squad_label(label)
                span_start, span_end = squad_utils.calc_tokenized_span_range(premise, hypothesis, answer,
                                                                             answer_start, answer_end,
                                                                             tokenizer, encoderModelType)
                span_start = seg_a_start + span_start
                span_end = min(seg_a_end, seg_a_start + span_end)
                answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[span_start:span_end])
                if span_start >= span_end:
                    span_start = -1
                    span_end = -1
                features = {
                    'uid': ids,
                    'label': is_impossible,
                    'answer': answer,
                    "answer_tokens": answer_tokens,
                    "token_start": span_start,
                    "token_end": span_end,
                    'token_id': input_ids,
                    'type_id': type_ids}
            else:
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))


def build_data_premise_and_multi_hypo(
        data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
    """Build QNLI as a pair-wise ranking task
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            hypothesis_1 = sample['hypothesis'][0]
            hypothesis_2 = sample['hypothesis'][1]
            label = sample['label']
            assert encoderModelType == EncoderModelType.BERT

            input_ids_1, _, type_ids_1 = bert_feature_extractor(
                premise, hypothesis_1, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
            input_ids_2, _, type_ids_2 = bert_feature_extractor(
                premise, hypothesis_2, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
            features = {
                'uid': ids, 'label': label, 'token_id': [
                    input_ids_1, input_ids_2], 'type_id': [
                    type_ids_1, type_ids_2], 'ruid': sample['ruid'], 'olabel': sample['olabel']}

            writer.write('{}\n'.format(json.dumps(features)))


def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly,
               max_seq_len=MAX_SEQ_LEN, encoderModelType=EncoderModelType.BERT, task_type=None, lab_dict=None):
    # We only support BERT based MRC for now
    if task_type == TaskType.Span:
        assert data_format == DataFormat.PremiseAndOneHypothesis
        assert encoderModelType == EncoderModelType.BERT
    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(data, dump_path, task_type, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.Sequence:
        build_data_sequence(data, dump_path, max_seq_len, tokenizer, lab_dict)
    else:
        raise ValueError(data_format)


def main(args):
    root = args['--root_dir']
    assert os.path.exists(root)

    max_seq_len = int(args['--max_seq_len'])
    log_file = os.path.join(root, 'blue_prepro_std_{}.log'.format(max_seq_len))
    logger = create_logger(__name__, to_disk=True, log_file=log_file)

    is_uncased = False
    if 'uncased' in args['--vocab']:
        is_uncased = True

    do_lower_case = args['--do_lower_case']
    mt_dnn_suffix = 'bert'
    encoder_model = EncoderModelType.BERT
    tokenizer = BertTokenizer.from_pretrained(args['--vocab'], do_lower_case=do_lower_case)

    if is_uncased:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = '{}_lower'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = BlueTaskDefs(args['--task_def'])

    if args['--datasets'] == 'all':
        tasks = task_defs.tasks
    else:
        tasks = args['--datasets'].split(',')
    for task in tasks:
        logger.info("Task %s" % task)
        data_format = task_defs.data_format_map[task]
        task_type = task_defs.task_type_map[task]
        label_mapper = task_defs.label_mapper_map[task]
        split_names = task_defs.split_names_map[task]
        for split_name in split_names:
            dump_path = os.path.join(mt_dnn_root, f"{task}_{split_name}.json")
            if os.path.exists(dump_path) and not args['--overwrite']:
                logger.warning('%s: Not overwrite %s: %s', task, split_name, dump_path)
                continue

            rows = load_data(os.path.join(root, f"{task}_{split_name}.tsv"), data_format, task_type, label_mapper)
            logger.info('%s: Loaded %s %s samples', task, len(rows), split_name)

            if task_type == TaskType.SequenceLabeling:
                rows = split_if_longer(rows, label_mapper, 30)
            build_data(rows, dump_path, tokenizer, data_format, max_seq_len=max_seq_len,
                       encoderModelType=encoder_model, task_type=task_type, lab_dict=label_mapper)

        logger.info('%s: Done', task)

    # combine train and dev
    for task in tasks:
        logger.info("Task %s: combine train and dev" % task)
        dump_path = os.path.join(mt_dnn_root, f"{task}_train+dev.json")
        if os.path.exists(dump_path) and not args['--overwrite']:
            logger.warning('%s: Not overwrite train+dev: %s', task, dump_path)
            continue
        task_type = task_defs.task_type_map[task]
        train_path = os.path.join(mt_dnn_root, f"{task}_train.json")
        train_rows = BatchGen.load(train_path, task_type=task_type)
        dev_path = os.path.join(mt_dnn_root, f"{task}_dev.json")
        dev_rows = BatchGen.load(dev_path, task_type=task_type)
        with open(dump_path, 'w', encoding='utf-8') as fp:
            for features in train_rows + dev_rows:
                fp.write('{}\n'.format(json.dumps(features)))
        logger.info('%s: Done', task)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args)
