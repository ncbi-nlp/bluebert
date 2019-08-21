#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# File Name: elmo_finetuning.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-03-29 19:27:12
###########################################################################
#

import os, sys, time, copy, pickle, logging, itertools
from collections import OrderedDict
from optparse import OptionParser
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn import metrics

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.seq2seq_encoders import FeedForwardEncoder, PytorchSeq2SeqWrapper, GatedCnnEncoder, IntraSentenceAttentionEncoder, QaNetEncoder, StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper, Seq2VecEncoder, CnnEncoder, CnnHighwayEncoder

import spacy
nlp = spacy.load('en_core_sci_md')


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'elmoft')
SC=';;'

opts, args = {}, []


class BaseClfHead(nn.Module):
    """ Classifier Head for the Basic Language Model """

    def __init__(self, lm_model, config, task_type, num_lbs=1, pdrop=0.1, mlt_trnsfmr=False, **kwargs):
        super(BaseClfHead, self).__init__()
        self.lm_model = lm_model
        self.task_type = task_type
        self.dropout = nn.Dropout2d(pdrop) if task_type == 'nmt' else nn.Dropout(pdrop)
        self.last_dropout = nn.Dropout(pdrop)
        self.lm_logit = self._mlt_lm_logit if mlt_trnsfmr else self._lm_logit
        self.clf_h = self._mlt_clf_h if mlt_trnsfmr else self._clf_h
        self.num_lbs = num_lbs
        self.kwprop = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _clf_h(self, hidden_states, pool_idx, past=None):
        return hidden_states.view(-1, self.n_embd), pool_idx

    def _mlt_clf_h(self, hidden_states, pool_idx, past=None):
        return hidden_states.sum(1).view(-1, self.n_embd), pool_idx.max(1)[0]

    def transformer(self, input_ids):
        return self.lm_model.transformer(input_ids=input_ids)

    def _lm_logit(self, input_ids, hidden_states, past=None):
        lm_h = hidden_states[:,:-1]
        return self.lm_model.lm_head(lm_h), input_ids[:,1:]

    def _mlt_lm_logit(self, input_ids, hidden_states, past=None):
        lm_h = hidden_states[:,:,:-1].contiguous().view(-1, self.n_embd)
        lm_target = input_ids[:,:,1:].contiguous().view(-1)
        return self.lm_model.lm_head(lm_h), lm_target.view(-1)


class ELMoClfHead(BaseClfHead):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, w2v_path=None, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, pool=None, seq2seq=None, seq2vec=None, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from gensim.models import KeyedVectors
        from gensim.models.keyedvectors import Word2VecKeyedVectors
        super(ELMoClfHead, self).__init__(lm_model, config, task_type, num_lbs=num_lbs, pdrop=pdrop, mlt_trnsfmr=mlt_trnsfmr, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        self.vocab_size = 793471
        self.dim_mulriple = 2 if task_type == 'entlmnt' or (task_type == 'sentsim' and (self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat')) else 1 # two or one sentence
        self.n_embd = config['elmoedim'] * 2 # two ELMo layer * sentence number * ELMo embedding dimensions
        self.w2v_model = w2v_path if type(w2v_path) is Word2VecKeyedVectors else (KeyedVectors.load(w2v_path, mmap='r') if w2v_path and os.path.isfile(w2v_path) else None)
        self._int_actvtn = ACTVTN_MAP[iactvtn]
        self._out_actvtn = ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.crf = ConditionalRandomField(num_lbs) if do_crf else None
        if task_type == 'nmt':
            self.pool = None
            self.seq2vec = None
            if seq2seq:
                params = {}
                if seq2seq.startswith('pytorch-'):
                    pth_mdl = '-'.join(seq2seq.split('-')[1:])
                    _ = [params.update(x) for x in [SEQ2SEQ_MDL_PARAMS.setdefault('pytorch', {}).setdefault('elmo', {}), SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(task_type, {})]]
                    self.seq2seq = gen_pytorch_wrapper('seq2seq', pth_mdl, **params[pth_mdl])
                    encoder_odim = SEQ2SEQ_DIM_INFER[seq2seq]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
                else:
                    _ = [params.update(x) for x in [SEQ2SEQ_MDL_PARAMS.setdefault(seq2seq, {}).setdefault('elmo', {}), SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(task_type, {})]]
                    self.seq2seq = SEQ2SEQ_MAP[seq2seq](**params)
                    if hasattr(self.seq2seq, 'get_output_dim'):
                        encoder_odim = self.seq2seq.get_output_dim()
                    else:
                        encoder_odim = SEQ2SEQ_DIM_INFER[seq2seq]([self.n_embd, self.dim_mulriple, params])
            else:
                self.seq2seq = None
                encoder_odim = self.n_embd
            self.maxlen = self.task_params.setdefault('maxlen', 128)
            self.norm = NORM_TYPE_MAP[norm_type](seqlen)
            self.linear = nn.Sequential(nn.Linear(encoder_odim, fchdim), self._int_actvtn(), nn.Linear(fchdim, fchdim), self._int_actvtn(), nn.Linear(fchdim, num_lbs), self._out_actvtn()) if fchdim else nn.Sequential(nn.Linear(encoder_odim, num_lbs), self._out_actvtn())
        elif seq2vec:
            self.pool = None
            params = {}
            if seq2vec.startswith('pytorch-'):
                pth_mdl = '-'.join(seq2vec.split('-')[1:])
                _ = [params.update(x) for x in [SEQ2VEC_MDL_PARAMS.setdefault('pytorch', {}).setdefault('elmo', {}), SEQ2VEC_TASK_PARAMS.setdefault('pytorch', {}).setdefault(task_type, {})]]
                _ = [params.update({p:kwargs[k]}) for k, p in SEQ2VEC_LM_PARAMS_MAP.setdefault('pytorch', []) if k in kwargs]
                self.seq2vec = gen_pytorch_wrapper('seq2vec', pth_mdl, **params[pth_mdl])
                encoder_odim = SEQ2VEC_DIM_INFER[seq2vec]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
            else:
                _ = [params.update(x) for x in [SEQ2VEC_MDL_PARAMS.setdefault(seq2vec, {}).setdefault('elmo', {}), SEQ2VEC_TASK_PARAMS.setdefault(seq2vec, {}).setdefault(task_type, {})]]
                _ = [params.update({p:kwargs[k]}) for k, p in SEQ2VEC_LM_PARAMS_MAP.setdefault(seq2vec, []) if k in kwargs]
                self.seq2vec = SEQ2VEC_MAP[seq2vec](**params)
                if hasattr(self.seq2vec, 'get_output_dim') and seq2vec != 'boe':
                    encoder_odim = self.seq2vec.get_output_dim()
                else:
                    encoder_odim = SEQ2VEC_DIM_INFER[seq2vec]([self.n_embd, self.dim_mulriple, params])
            self.maxlen = self.task_params.setdefault('maxlen', 128)
            self.norm = NORM_TYPE_MAP[norm_type](encoder_odim)
            self.linear = (nn.Sequential(nn.Linear(self.dim_mulriple * encoder_odim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(encoder_odim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.dim_mulriple * encoder_odim, self.dim_mulriple * encoder_odim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.dim_mulriple * encoder_odim, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(encoder_odim, num_lbs))
        elif pool:
            self.seq2vec = None
            self.pool = nn.MaxPool2d(8, stride=4) if pool == 'max' else nn.AvgPool2d(8, stride=4)
            self.norm = NORM_TYPE_MAP[norm_type](32130 if self.task_type == 'sentsim' or self.task_type == 'entlmnt' else 16065)
            self.linear = (nn.Sequential(nn.Linear(self.dim_mulriple * encoder_odim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(encoder_odim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.dim_mulriple * encoder_odim, self.dim_mulriple * encoder_odim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.dim_mulriple * encoder_odim, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(encoder_odim, num_lbs))
        else:
            self.pool = None
            self.seq2vec = None
            self.norm = NORM_TYPE_MAP[norm_type](self.n_embd)
            self.linear = (nn.Sequential(nn.Linear(self.n_embd, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.n_embd, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, num_lbs))) if fchdim else (nn.Sequential(*([nn.Linear(self.n_embd, self.n_embd), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.n_embd, num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.n_embd, num_lbs))
        if (initln): self.linear.apply(_weights_init(mean=initln_mean, std=initln_std))

    def forward(self, input_ids, pool_idx, w2v_ids=None, char_ids=None, labels=None, past=None, weights=None):
        use_gpu = next(self.parameters()).is_cuda
        if self.task_type in ['entlmnt', 'sentsim']:
            mask = [torch.arange(input_ids[x].size()[1]).to('cuda').unsqueeze(0).expand(input_ids[x].size()[:2]) <= pool_idx[x].unsqueeze(1).expand(input_ids[x].size()[:2]) if use_gpu else torch.arange(input_ids[x].size()[1]).unsqueeze(0).expand(input_ids[x].size()[:2]) <= pool_idx[x].unsqueeze(1).expand(input_ids[x].size()[:2]) for x in [0,1]]
            embeddings = (self.lm_model(input_ids[0]), self.lm_model(input_ids[1]))
            clf_h = torch.cat(embeddings[0]['elmo_representations'], dim=-1), torch.cat(embeddings[1]['elmo_representations'], dim=-1)
            if (w2v_ids is not None and self.w2v_model):
                wembd_tnsr = [torch.tensor([self.w2v_model.syn0[s] for s in w2v_ids[x]]) for x in [0,1]]
                if use_gpu: wembd_tnsr = [x.to('cuda') for x in wembd_tnsr]
                clf_h = [torch.cat([clf_h[x], wembd_tnsr[x]], dim=-1) for x in [0,1]]
            if self.seq2vec:
                clf_h = [self.seq2vec(clf_h[x], mask=mask[x]) for x in [0,1]]
            elif self.pool:
                clf_h = [clf_h[x].view(clf_h[x].size(0), 2*clf_h[x].size(1), -1) for x in [0,1]]
                clf_h = [self.pool(clf_h[x]).view(clf_h[x].size(0), -1) for x in [0,1]]
            else:
                clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]]
        else:
            mask = torch.arange(input_ids.size()[1]).to('cuda').unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2]) if use_gpu else torch.arange(input_ids.size()[1]).unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2])
            embeddings = self.lm_model(input_ids)
            clf_h = torch.cat(embeddings['elmo_representations'], dim=-1)
            if self.task_type == 'nmt':
                clf_h = clf_h
                if (self.seq2seq): clf_h = self.seq2seq(clf_h, mask=mask)
            elif self.seq2vec:
                clf_h = self.seq2vec(clf_h, mask=mask)
            elif self.pool:
                clf_h = clf_h.view(clf_h.size(0), 2*clf_h.size(1), -1)
                clf_h = self.pool(clf_h).view(clf_h.size(0), -1)
            else:
                clf_h = clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        if self.task_type in ['entlmnt', 'sentsim']:
            if self.do_norm: clf_h = [self.norm(clf_h[x]) for x in [0,1]]
            clf_h = [self.dropout(clf_h[x]) for x in [0,1]]
            if (self.task_type == 'entlmnt' or self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat'):
                # clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
                clf_h = torch.cat(clf_h, dim=-1)
                clf_logits = self.linear(clf_h) if self.linear else clf_h
            else:
                clf_logits = clf_h = F.pairwise_distance(self.linear(clf_h[0]), self.linear(clf_h[1]), 2, eps=1e-12) if self.task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.linear(clf_h[0]), self.linear(clf_h[1]), dim=1, eps=1e-12)
        else:
            if self.do_norm: clf_h = self.norm(clf_h)
            clf_h = self.dropout(clf_h)
            clf_logits = self.linear(clf_h)
            if self.do_lastdrop: clf_logits = self.last_dropout(clf_logits)

        if (labels is None):
            if self.crf:
                tag_seq, score = zip(*self.crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), torch.ones(*(input_ids.size()[:2])).int()))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                clf_logits = torch.zeros((*tag_seq.size(), self.num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits
            if (self.task_type == 'sentsim' and self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != self.task_params['ymode']): return 1 - clf_logits.view(-1, self.num_lbs)
            return clf_logits.view(-1, self.num_lbs)
        if self.crf:
            clf_loss = -self.crf(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), mask.long())
        elif self.task_type == 'mltc-clf' or self.task_type == 'entlmnt' or self.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1))
        elif self.task_type == 'mltl-clf':
            loss_func = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1, self.num_lbs).float())
        elif self.task_type == 'sentsim':
            loss_func = ContrastiveLoss(reduction='none', x_mode=SIM_FUNC_MAP.setdefault(self.task_params['sentsim_func'], 'dist'), y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else nn.MSELoss(reduction='none')
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        return clf_loss, None



class BaseDataset(Dataset):
    """Basic dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], mltl=False, **kwargs):
        self.text_col = [str(s) for s in text_col] if hasattr(text_col, '__iter__') and type(text_col) is not str else str(text_col)
        self.label_col = [str(s) for s in label_col] if hasattr(label_col, '__iter__') and type(label_col) is not str else str(label_col)
        self.df = self._df = csv_file if type(csv_file) is pd.DataFrame else pd.read_csv(csv_file, sep=sep, encoding='utf-8', engine='python', error_bad_lines=False, dtype={self.label_col:'float' if binlb == 'rgrsn' else str}, **kwargs)
        self.df.columns = self.df.columns.astype(str, copy=False)
        self.df = self.df[self.df[self.label_col].notnull()]
        self.mltl = mltl
        if (binlb == 'rgrsn'):
            self.binlb = None
            self.binlbr = None
        elif (type(binlb) is str and binlb.startswith('mltl')):
            sc = binlb.split(SC)[-1]
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set([lb for lbs in lb_df for lb in lbs.split(sc)])) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.binlbr = OrderedDict([(i, lb) for i, lb in enumerate(labels)])
            self.mltl = True
        elif (binlb is None):
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set(lb_df)) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.binlbr = OrderedDict([(i, lb) for i, lb in enumerate(labels)])
        else:
            self.binlb = binlb
            self.binlbr = OrderedDict([(i, lb) for lb, i in binlb.items()])
        self.encode_func = encode_func
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer), record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0]) is str or type(sample[0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1])

    def _transform_chain(self, sample):
        if self.transforms:
            self.transforms = self.transforms if type(self.transforms) is list else [self.transforms]
            self.transforms_kwargs = self.transforms_kwargs if type(self.transforms_kwargs) is list else [self.transforms_kwargs]
            for transform, transform_kwargs in zip(self.transforms, self.transforms_kwargs):
                transform_kwargs.update(self.transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self, transform)(sample, **transform_kwargs)
        return sample

    def _nmt_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], [self.binlb.setdefault(y, len(self.binlb)) for y in sample[1]]

    def _mltc_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], self.binlb.setdefault(sample[1], len(self.binlb))

    def _mltl_transform(self, sample, options=None, binlb={}, get_lb=lambda x: x.split(SC)):
        if (len(binlb) > 0): self.binlb = binlb
        labels = get_lb(sample[1])
        return sample[0], [1 if lb in labels else 0 for lb in self.binlb.keys()]

    def fill_labels(self, lbs, binlb=True, index=None, saved_path=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [(';'.join([self.binlbr[l] for l in np.where(lb == 1)[0]]) if self.mltl else ','.join(['_'.join([str(i), str(l)]) for i, l in enumerate(lb)])) if hasattr(lb, '__iter__') else self.binlbr[lb] for lb in lbs]
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        try:
            if index:
                filled_df.loc[index, self.label_col] = lbs
            else:
                filled_df[self.label_col] = lbs
        except Exception as e:
            print(e)
            with open('pred_lbs.tmp', 'wb') as fd:
                pickle.dump((filled_df, index, self.label_col, lbs), fd)
            raise e
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df

    def rebalance(self):
        if (self.binlb is None): return
        task_cols, task_trsfm, task_extparms = TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        if (type(lb_df.iloc[0]) is list):
            lb_df[:] = [self._mltl_transform((None, SC.join(lbs)))[1] for lbs in lb_df]
            max_lb_df = lb_df.loc[[idx for idx, lbs in lb_df.iteritems() if np.sum(list(map(int, lbs))) == 0]]
            max_num, avg_num = max_lb_df.shape[0], 1.0 * lb_df[~lb_df.index.isin(max_lb_df.index)].shape[0] / len(lb_df.iloc[0])
        else:
            class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y is not None]).sum(axis=0)
            max_num, max_lb_bin = class_count.max(), class_count.argmax()
            max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
            avg_num = np.mean([class_count[x] for x in range(len(class_count)) if x != max_lb_bin])
        removed_idx = max_lb_df.sample(n=int(max_num-avg_num), random_state=1).index
        self.df = self.df.loc[list(set(self.df.index)-set(removed_idx))]

    def remove_mostfrqlb(self):
        if (self.binlb is None or self.binlb == 'rgrsn'): return
        task_cols, task_trsfm, task_extparms = TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y]).sum(axis=0)
        max_num, max_lb_bin = class_count.max(), class_count.argmax()
        max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
        self.df = self.df.loc[list(set(self.df.index)-set(max_lb_df.index))]


class SentSimDataset(BaseDataset):
    """Sentence Similarity task dataset class"""

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0][0]) is str or type(sample[0][0][0]) is str else torch.tensor(sample[0])), torch.tensor(0 if sample[1] is np.nan else float(sample[1]) / 5.0)

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        lbs = 5.0 * lbs
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        if index:
            filled_df.loc[index, self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df


class EntlmntDataset(BaseDataset):
    """Entailment task dataset class"""

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0][0]) is str or (type(sample[0][0]) is list and type(sample[0][0][0]) is str) else torch.tensor(sample[0])), torch.tensor(sample[1])


class NERDataset(BaseDataset):
    """NER task dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], **kwargs):
        super(NERDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, sep=sep, header=None, skip_blank_lines=False, keep_default_na=False, na_values=[], binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, **kwargs)
        sep_selector = self.df[self.text_col].apply(lambda x: True if x=='.' else False)
        sep_selector.iloc[-1] = True
        int_idx = pd.DataFrame(np.arange(self.df.shape[0]), index=self.df.index)
        self.boundaries = [0] + list(itertools.chain.from_iterable((int_idx[sep_selector.values].values+1).tolist()))

    def __len__(self):
        return len(self.boundaries) - 1

    def __getitem__(self, idx):
        record = self.df.iloc[self.boundaries[idx]:self.boundaries[idx+1]].dropna()
        sample = self.encode_func(record[self.text_col].values.tolist(), self.tokenizer), record[self.label_col].values.tolist()
        num_samples = [len(x) for x in sample[0]] if (len(sample[0]) > 0 and type(sample[0][0]) is list) else [1] * len(sample[0])
        record_idx = [0] + np.cumsum(num_samples).tolist()
        is_empty = (type(sample[0]) is list and len(sample[0]) == 0) or (type(sample[0]) is list and len(sample[0]) > 0 and all([type(x) is list and len(x) == 0 for x in sample[0]]))
        if (is_empty): return SC.join(map(str, record.index.values.tolist())), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*opts.maxlen), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*opts.maxlen), SC.join(map(str, record_idx))
        is_encoded = (type(sample[0]) is list and type(sample[0][0]) is int) or (type(sample[0]) is list and len(sample[0]) > 0 and type(sample[0][0]) is list and len(sample[0][0]) > 0 and type(sample[0][0][0]) is int)
        sample = list(itertools.chain.from_iterable(sample[0])) if is_encoded else sample[0], list(itertools.chain.from_iterable([[x] * ns for x, ns in zip(sample[1], num_samples)]))
        sample = self._transform_chain(sample)
        return SC.join(map(str, record.index.values.tolist())), (torch.tensor(sample[0]) if is_encoded else SC.join(sample[0])), (torch.tensor(sample[1]) if is_encoded else SC.join(map(str, sample[1]))), SC.join(map(str, record_idx))

    def fill_labels(self, lbs, saved_path=None, binlb=True, index=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [self.binlbr[lb] for lb in lbs]
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        if index:
            filled_df[self.label_col] = ''
            filled_df.loc[index, self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, sep='\t', header=None, index=None, **kwargs)
        return filled_df


def _sentclf_transform(sample, options=None, start_tknids=[], clf_tknids=[]):
    X, y = sample
    X = [start_tknids + x + clf_tknids for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X + clf_tknids
    return X, y


def _entlmnt_transform(sample, options=None, start_tknids=[], clf_tknids=[], delim_tknids=[]):
    X, y = sample
    X = start_tknids + X[0] + delim_tknids + X[1] + clf_tknids
    return X, y


def _sentsim_transform(sample, options=None, start_tknids=[], clf_tknids=[], delim_tknids=[]):
    X, y = sample
    X = [start_tknids + X[0] + delim_tknids + X[1] + clf_tknids, start_tknids + X[1] + delim_tknids + X[0] + clf_tknids]
    return X, y


def _padtrim_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None):
    X, y = sample
    X = [x[:min(seqlen, len(x))] + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))] + [xpad_val] * (seqlen - len(X))
    if ypad_val is not None: y = [x[:min(seqlen, len(x))] + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))] + [ypad_val] * (seqlen - len(y))
    return X, y


def _trim_transform(sample, options=None, seqlen=32, trimlbs=False, special_tkns={}):
    seqlen -= sum([len(v) for v in special_tkns.values()])
    X, y = sample
    X = [x[:min(seqlen, len(x))] for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))]
    if trimlbs: y = [x[:min(seqlen, len(x))] for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))]
    return X, y


def _pad_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None):
    X, y = sample
    X = [x + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X + [xpad_val] * (seqlen - len(X))
    if ypad_val is not None: y = [x + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y + [ypad_val] * (seqlen - len(y))
    return X, y


def _adjust_encoder(mdl_name, tokenizer, extra_tokens=[], ret_list=False):
    return [[tkn] if ret_list else tkn for tkn in extra_tokens]


def _tokenize(text, tokenizer):
    return text


def _weights_init(mean=0., std=0.02):
    def _wi(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(mean, std)
        elif classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.normal_(m.bias, 0)
    return _wi


def elmo_config(options_path, weights_path, elmoedim=1024, dropout=0.5):
    return {'options_file':options_path, 'weight_file':weights_path, 'num_output_representations':2, 'elmoedim':elmoedim, 'dropout':dropout}

TASK_TYPE_MAP = {'bc5cdr-chem':'nmt', 'bc5cdr-dz':'nmt', 'shareclefe':'nmt', 'ddi':'mltc-clf', 'chemprot':'mltc-clf', 'i2b2':'mltc-clf', 'hoc':'mltl-clf', 'mednli':'entlmnt', 'biosses':'sentsim', 'clnclsts':'sentsim'}
TASK_PATH_MAP = {'bc5cdr-chem':'BC5CDR-chem', 'bc5cdr-dz':'BC5CDR-disease', 'shareclefe':'ShAReCLEFEHealthCorpus', 'ddi':'ddi2013-type', 'chemprot':'ChemProt', 'i2b2':'i2b2-2010', 'hoc':'hoc', 'mednli':'mednli', 'biosses':'BIOSSES', 'clnclsts':'clinicalSTS'}
TASK_DS_MAP = {'bc5cdr-chem':NERDataset, 'bc5cdr-dz':NERDataset, 'shareclefe':NERDataset, 'ddi':BaseDataset, 'chemprot':BaseDataset, 'i2b2':BaseDataset, 'hoc':BaseDataset, 'mednli':EntlmntDataset, 'biosses':SentSimDataset, 'clnclsts':SentSimDataset}
TASK_COL_MAP = {'bc5cdr-chem':{'index':False, 'X':'0', 'y':'3'}, 'bc5cdr-dz':{'index':False, 'X':'0', 'y':'3'}, 'shareclefe':{'index':False, 'X':'0', 'y':'3'}, 'ddi':{'index':'index', 'X':'sentence', 'y':'label'}, 'chemprot':{'index':'index', 'X':'sentence', 'y':'label'}, 'i2b2':{'index':'index', 'X':'sentence', 'y':'label'}, 'hoc':{'index':'index', 'X':'sentence', 'y':'labels'}, 'mednli':{'index':'id', 'X':['sentence1','sentence2'], 'y':'label'}, 'biosses':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}, 'clnclsts':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}}
TASK_TRSFM = {'bc5cdr-chem':(['_nmt_transform'], [{}]), 'bc5cdr-dz':(['_nmt_transform'], [{}]), 'shareclefe':(['_nmt_transform'], [{}]), 'ddi':(['_mltc_transform'], [{}]), 'chemprot':(['_mltc_transform'], [{}]), 'i2b2':(['_mltc_transform'], [{}]), 'hoc':(['_mltl_transform'], [{ 'get_lb':lambda x: [s.split('_')[0] for s in x.split(',') if s.split('_')[1] == '1'], 'binlb': dict([(str(x),x) for x in range(10)])}]), 'mednli':(['_mltc_transform'], [{}]), 'biosses':([], []), 'clnclsts':([], [])}
TASK_EXT_TRSFM = {'bc5cdr-chem':([_padtrim_transform], [{}]), 'bc5cdr-dz':([_padtrim_transform], [{}]), 'shareclefe':([_padtrim_transform], [{}]), 'ddi':([_trim_transform, _sentclf_transform, _pad_transform], [{},{},{}]), 'chemprot':([_trim_transform, _sentclf_transform, _pad_transform], [{},{},{}]), 'i2b2':([_trim_transform, _sentclf_transform, _pad_transform], [{},{},{}]), 'hoc':([_trim_transform, _sentclf_transform, _pad_transform], [{},{},{}]), 'mednli':([_trim_transform, _entlmnt_transform, _pad_transform], [{},{},{}]), 'biosses':([_trim_transform, _sentsim_transform, _pad_transform], [{},{},{}]), 'clnclsts':([_trim_transform, _sentsim_transform, _pad_transform], [{},{},{}])}
TASK_EXT_PARAMS = {'bc5cdr-chem':{'ypad_val':'O', 'trimlbs':True, 'mdlcfg':{'maxlen':128}}, 'bc5cdr-dz':{'ypad_val':'O', 'trimlbs':True, 'mdlcfg':{'maxlen':128}}, 'shareclefe':{'ypad_val':'O', 'trimlbs':True, 'mdlcfg':{'maxlen':128}}, 'ddi':{'mdlcfg':{'maxlen':128}}, 'chemprot':{'mdlcfg':{'maxlen':128}}, 'i2b2':{'mdlcfg':{'maxlen':128}}, 'hoc':{'binlb': OrderedDict([(str(x),x) for x in range(10)]), 'mdlcfg':{'maxlen':128}}, 'mednli':{'mdlcfg':{'maxlen':128}}, 'biosses':{'binlb':'rgrsn', 'mdlcfg':{'maxlen':128}}, 'clnclsts':{'binlb':'rgrsn', 'ymode':'sim', 'mdlcfg':{'sentsim_func':None, 'maxlen':128}}}

MDL_NAME_MAP = {'elmo':'elmo'}
PARAMS_MAP = {'elmo':'ELMo'}
ENCODE_FUNC_MAP = {'elmo':_tokenize}
MODEL_MAP = {'elmo':Elmo}
CLF_MAP = {'elmo':ELMoClfHead}
CLF_EXT_PARAMS = {'elmo':{'pool':False, 'seq2seq':'isa', 'seq2vec':'boe', 'fchdim':768, 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02}}
CONFIG_MAP = {'elmo':elmo_config}
TKNZR_MAP = {'elmo':None}
PYTORCH_WRAPPER = {'lstm':nn.LSTM, 'rnn':nn.RNN, 'gru':nn.GRU, 'agmnlstm':AugmentedLstm, 'stkaltlstm':StackedAlternatingLstm}
SEQ2SEQ_MAP = {'ff':FeedForwardEncoder, 'pytorch':PytorchSeq2SeqWrapper, 'cnn':GatedCnnEncoder, 'isa':IntraSentenceAttentionEncoder, 'qanet':QaNetEncoder, 'ssae':StackedSelfAttentionEncoder}
SEQ2SEQ_MDL_PARAMS = {'pytorch':{'elmo':{'lstm':{'input_size':2048,'hidden_size':768, 'batch_first':True}, 'rnn':{'input_size':2048,'hidden_size':768, 'batch_first':True}, 'gru':{'input_size':2048,'hidden_size':768, 'batch_first':True},'agmnlstm':{'input_size':2048,'hidden_size':768},'stkaltlstm':{'input_size':2048,'hidden_size':768, 'num_layers':3}}}, 'cnn':{'elmo':{'input_dim':2048, 'dropout':0.5, 'layers':[[[4, 2048]],[[4, 2048],[4, 2048]]]}}, 'isa':{'elmo':{'input_dim':2048}}, 'qanet':{'elmo':{}}, 'ssae':{'elmo':{'input_dim':2048, 'hidden_dim':1024, 'projection_dim':768, 'feedforward_hidden_dim':768, 'num_layers':1, 'num_attention_heads':8}}}
SEQ2SEQ_TASK_PARAMS = {}
SEQ2VEC_MAP = {'boe':BagOfEmbeddingsEncoder, 'pytorch':PytorchSeq2VecWrapper, 'allennlp':Seq2VecEncoder, 'cnn':CnnEncoder, 'cnn_highway':CnnHighwayEncoder}
SEQ2VEC_MDL_PARAMS = { \
	'boe':{ \
		'elmo':{'embedding_dim':768, 'averaged':True} \
	}, \
	'pytorch':{ \
		'elmo':{ \
			'lstm':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'rnn':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'gru':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'agmnlstm':{'input_size':2048,'hidden_size':768}, \
			'stkaltlstm':{'input_size':2048,'hidden_size':768, 'num_layers':3} \
		} \
	}, \
	'cnn':{ \
		'elmo':{'embedding_dim':2048, 'num_filters':768} \
	}, \
	'cnn_highway':{ \
		'elmo':{'embedding_dim':2048, 'filters':[(2, 768),(3, 768),(4, 768),(5, 768)], 'num_highway':5, 'projection_dim':2048} \
	} \
}
SEQ2VEC_TASK_PARAMS = {}
SEQ2VEC_LM_PARAMS_MAP = {'boe':[('hdim','embedding_dim')], 'pytorch':[('hdim', 'hidden_size')], 'cnn':[], 'cnn_highway':[]}
SEQ2SEQ_DIM_INFER = {'pytorch-lstm':lambda x: x[1] * x[2]['hidden_size'], 'pytorch-rnn':lambda x: x[1] * x[2]['hidden_size'], 'pytorch-gru':lambda x: x[1] * x[2]['hidden_size'], 'cnn':lambda x: 2 * x[0], 'isa':lambda x: x[0]}
SEQ2VEC_DIM_INFER = {'boe':lambda x: x[0], 'pytorch-lstm':lambda x: x[2]['hidden_size'], 'pytorch-agmnlstm':lambda x: x[2]['hidden_size'], 'pytorch-rnn':lambda x: x[2]['hidden_size'], 'pytorch-stkaltlstm':lambda x: x[2]['hidden_size'], 'pytorch-gru':lambda x: x[2]['hidden_size'], 'cnn':lambda x: int(1.5 * x[2]['embedding_dim']), 'cnn_highway':lambda x: x[0]}
NORM_TYPE_MAP = {'batch':nn.BatchNorm1d, 'layer':nn.LayerNorm}
ACTVTN_MAP = {'relu':nn.ReLU, 'sigmoid':nn.Sigmoid}
SIM_FUNC_MAP = {'sim':'sim', 'dist':'dist'}

LM_PARAMS = {
    "ELMo": {
        "options_path": "options.json",
        "weights_path": "weights.hdf5",
        "elmoedim": 1024,
        "dropout": 0.5
    }
}


def gen_mdl(mdl_name, pretrained=True, use_gpu=False, distrb=False, dev_id=None):
    try:
        params = LM_PARAMS[PARAMS_MAP[mdl_name]]
        config = CONFIG_MAP[mdl_name](**params)
        pos_params = [config[k] for k in ['options_file','weight_file', 'num_output_representations']]
        kw_params = dict([(k, config[k]) for k in config.keys() if k not in ['options_file','weight_file', 'num_output_representations', 'elmoedim']])
        model = MODEL_MAP[mdl_name](*pos_params, **kw_params)
    except Exception as e:
        print(e)
        print('Cannot find the pretrained model file, using online model instead.')
        model = MODEL_MAP[mdl_name].from_pretrained(MDL_NAME_MAP[mdl_name])
    if (use_gpu): model = _handle_model(model, dev_id=dev_id, distrb=distrb)
    return model


def gen_clf(mdl_name, use_gpu=False, distrb=False, dev_id=None, **kwargs):
    params = LM_PARAMS[PARAMS_MAP[mdl_name]]
    kwargs['config'] = CONFIG_MAP[mdl_name](**params)
    clf = CLF_MAP[mdl_name](**kwargs)
    return clf.to('cuda') if use_gpu else clf


def classify(dev_id=None):
    use_gpu = dev_id is not None
    encode_func = ENCODE_FUNC_MAP[opts.model]
    tokenizer = None
    task_type = TASK_TYPE_MAP[opts.task]
    special_tkns = (['start_tknids', 'delim_tknids', 'clf_tknids'], ['_@_', ' _#_', ' _$_']) if task_type == 'sentsim' else (['start_tknids', 'clf_tknids'], ['_@_', ' _$_'])
    special_tknids = _adjust_encoder(opts.model, None, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))

    # Prepare data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = TASK_PATH_MAP[opts.task], TASK_DS_MAP[opts.task], TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
    trsfms = task_trsfm[0] if len(task_trsfm) > 0 else []
    trsfms_kwargs = task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm)

    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if (task_type == 'sentsim'):
        class_count = None
    elif len(lb_trsfm) > 0:
        lb_df = train_ds.df[task_cols['y']].apply(lb_trsfm[0])
        class_count = np.array([[1 if lb in y else 0 for lb in task_extparms.setdefault('binlb', train_ds.binlb).keys()] for y in lb_df]).sum(axis=0)
    else:
        lb_df = train_ds.df[task_cols['y']]
        binlb = task_extparms.setdefault('binlb', train_ds.binlb)
        class_count = lb_df.value_counts()[binlb.keys()].values
    if (class_count is None):
        class_weights = None
        sampler = None
    else:
        class_weights = torch.Tensor(1.0 / class_count)
        class_weights /= class_weights.sum()
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=opts.bsize, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=opts.bsize, shuffle=False, sampler=None, num_workers=opts.np, drop_last=opts.droplast)

    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    # Load model
    mdl_name = opts.model.lower().replace(' ', '_')
    if (opts.resume):
        clf = load_model(opts.resume)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
    else:
        # Build model
        lm_model = gen_mdl(opts.model, pretrained=True if type(opts.pretrained) is str and opts.pretrained.lower() == 'true' else opts.pretrained, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id)
        clf = gen_clf(opts.model, lm_model=lm_model, task_type=task_type, num_lbs=len(train_ds.binlb) if train_ds.binlb else 1, mlt_trnsfmr=True if task_type=='sentsim' else False, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id, **dict([(k, getattr(opts, k)) if hasattr(opts, k) else (k, v) for k, v in CLF_EXT_PARAMS.setdefault(opts.model, {}).items()]))
        # optimizer = torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        optimizer = torch.optim.Adam(clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay)

        # Training
        train(clf, optimizer, train_loader, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, task_type=task_type, task_name=opts.task, mdl_name=mdl_name, use_gpu=use_gpu, devq=dev_id)

    # Evaluation
    eval(clf, dev_loader, dev_ds.binlbr, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=mdl_name, use_gpu=use_gpu)
    eval(clf, test_loader, test_ds.binlbr, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=mdl_name, use_gpu=use_gpu)


def train(clf, optimizer, dataset, clf_tknids, pad_val=0, weights=None, lmcoef=0.5, clipmaxn=0.25, epochs=1, task_type='mltc-clf', task_name='classification', mdl_name='sota', use_gpu=False, devq=None):
    clf.train()
    for epoch in range(epochs):
        total_loss = 0
        if task_type != 'entlmnt' and task_type != 'sentsim': dataset.dataset.rebalance()
        for step, batch in enumerate(tqdm(dataset, desc='[%i/%i epoch(s)] Training batches' % (epoch + 1, epochs))):
            optimizer.zero_grad()
            if task_type == 'nmt':
                idx, tkns_tnsr, lb_tnsr, record_idx = batch
                record_idx = [list(map(int, x.split(SC))) for x in record_idx]
            else:
                idx, tkns_tnsr, lb_tnsr = batch
            if task_type == 'entlmnt' or task_type == 'sentsim':
                tkns_tnsr = [[[w.text for w in nlp(sents)] for sents in tkns_tnsr[x]] for x in [0,1]]
                tkns_tnsr = [[s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
                tkns_tnsr = [batch_to_ids(tkns_tnsr[x]) for x in [0,1]]
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]], (weights if weights is None else weights.to('cuda'))
            elif task_type == 'nmt':
                tkns_tnsr, lb_tnsr = [s.split(SC) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(SC))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
                if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
                tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                lb_tnsr = torch.LongTensor([s[:min(len(s), opts.maxlen)] + [pad_val] * (opts.maxlen-len(s)) for s in lb_tnsr])
                pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                tkns_tnsr = batch_to_ids(tkns_tnsr)
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda'))
            else:
                tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
                if clf.pool: tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                tkns_tnsr = batch_to_ids(tkns_tnsr)
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda'))
            clf_loss, lm_loss = clf(input_ids=tkns_tnsr, pool_idx=pool_idx, labels=lb_tnsr.view(-1), weights=weights)
            train_loss = clf_loss.mean() if lm_loss is None else (clf_loss.mean() + lmcoef * ((lm_loss.view(tkns_tnsr.size(0), -1) * mask_tnsr).sum(1) / (1e-12 + mask_tnsr.sum(1))).mean())
            total_loss += train_loss.item()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), clipmaxn)
            train_loss.backward()
            optimizer.step()
        print('Train loss in %i epoch(s): %f' % (epoch + 1, total_loss / (step + 1)))
    save_model(clf, optimizer, '%s_%s.pth' % (task_name, mdl_name), devq=devq)


def eval(clf, dataset, binlbr, clf_tknids, pad_val=0, task_type='mltc-clf', task_name='classification', ds_name='', mdl_name='sota', clipmaxn=0.25, use_gpu=False):
    clf.eval()
    total_loss, indices, preds, probs, all_logits, trues, ds_name = 0, [], [], [], [], [], ds_name.strip()
    if task_type not in ['entlmnt', 'sentsim', 'mltl-clf']: dataset.dataset.remove_mostfrqlb()
    for step, batch in enumerate(tqdm(dataset, desc="%s batches" % ds_name.title() if ds_name else 'Evaluation')):
        if task_type == 'nmt':
            idx, tkns_tnsr, lb_tnsr, record_idx = batch
            record_idx = [list(map(int, x.split(SC))) for x in record_idx]
        else:
            idx, tkns_tnsr, lb_tnsr = batch
        indices.extend(idx if type(idx) is list else (idx.tolist() if type(idx) is torch.Tensor else list(idx)))
        _lb_tnsr = lb_tnsr
        if task_type == 'entlmnt' or task_type == 'sentsim':
            tkns_tnsr = [[[w.text for w in nlp(sents)] for sents in tkns_tnsr[x]] for x in [0,1]]
            tkns_tnsr = [[s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
            pool_idx = _pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
            tkns_tnsr = [batch_to_ids(tkns_tnsr[x]) for x in [0,1]]
            if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx= [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]]
        elif task_type == 'nmt':
            # tkns_tnsr, lb_tnsr = [s.split(SC) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(SC))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
            tkns_tnsr, lb_tnsr = zip(*[(sx.split(SC), list(map(int, sy.split(SC)))) for sx, sy in zip(tkns_tnsr, lb_tnsr) if ((type(sx) is str and sx != '') or len(sx) > 0) and ((type(sy) is str and sy != '') or len(sy) > 0)])
            if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
            tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
            _lb_tnsr = lb_tnsr = torch.LongTensor([s[:min(len(s), opts.maxlen)] + [pad_val] * (opts.maxlen-len(s)) for s in lb_tnsr])
            pool_idx = _pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
            tkns_tnsr = batch_to_ids(tkns_tnsr)
            if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda')
        else:
            tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
            if clf.pool: tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
            pool_idx = _pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
            tkns_tnsr = batch_to_ids(tkns_tnsr)
            if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx= tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda')
        with torch.no_grad():
            logits = clf(tkns_tnsr, pool_idx, labels=None)
            if task_type == 'mltc-clf' or task_type == 'entlmnt':
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
                prob, pred = torch.softmax(logits, -1).max(-1)
            elif task_type == 'mltl-clf':
                loss_func = nn.BCEWithLogitsLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1, len(binlbr)).float())
                prob = torch.sigmoid(logits).data
                pred = (prob > opts.pthrshld).int()
            elif task_type == 'nmt':
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
                prob, pred = torch.softmax(logits, -1).max(-1)
            elif task_type == 'sentsim':
                loss_func = nn.MSELoss(reduction='none')
                loss = loss_func(logits.view(-1), lb_tnsr.view(-1))
                prob, pred = logits, logits
            total_loss += loss.mean().item()
        if task_type == 'nmt':
            last_tkns = torch.arange(_lb_tnsr.size(0)) * _lb_tnsr.size(1) + _pool_idx
            flat_tures, flat_preds, flat_probs = _lb_tnsr.view(-1).tolist(), pred.view(-1).detach().cpu().tolist(), prob.view(-1).detach().cpu().tolist()
            flat_tures_set, flat_preds_set, flat_probs_set = set(flat_tures), set(flat_preds), set(flat_probs)
            trues.append([[max(flat_tures_set, key=flat_tures[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
            preds.append([[max(flat_preds_set, key=flat_preds[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
            probs.append([[max(flat_probs_set, key=flat_probs[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
        else:
            trues.append(_lb_tnsr.view(_lb_tnsr.size(0), -1).numpy() if task_type == 'mltl-clf' else _lb_tnsr.view(-1).detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

        all_logits.append(logits.view(_lb_tnsr.size(0), -1, logits.size(-1)).detach().cpu().numpy())
    total_loss = total_loss / (step + 1)
    print('Evaluation loss on %s dataset: %.2f' % (ds_name, total_loss))

    all_logits = np.concatenate(all_logits, axis=0)
    if task_type == 'nmt':
        trues = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(trues))))
        preds = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(preds))))
        probs = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(probs))))
    else:
        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0)
    resf_prefix = ds_name.lower().replace(' ', '_')
    with open('%s_preds_trues.pkl' % resf_prefix, 'wb') as fd:
        pickle.dump((trues, preds, probs, all_logits), fd)
    if any(task_type == t for t in ['mltc-clf', 'entlmnt', 'nmt']):
        preds = preds
    elif task_type == 'mltl-clf':
        preds = preds
    elif task_type == 'sentsim':
        preds = np.squeeze(preds)
    if task_type == 'sentsim':
        if (np.isnan(preds).any()):
            print('Predictions contain NaN values! Please try to decrease the learning rate!')
            return
        metric_names, metrics_funcs = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Squared Log Error', 'Median Absolute Error', 'R2', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.median_absolute_error, metrics.r2_score, _prsn_cor]
        perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[mdl_name])[metric_names]
    elif task_type == 'mltl-clf':
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, target_names=[binlbr[x] for x in binlbr.keys()], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    else:
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, target_names=[binlbr[x] for x in binlbr.keys() if x in preds or x in trues], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    print('Results for %s dataset is:\n%s' % (ds_name.title(), perf_df))
    perf_df.to_excel('perf_%s.xlsx' % resf_prefix)
    if (type(indices[0]) is str and SC in indices[0]):
        indices = list(itertools.chain.from_iterable([list(map(int, idx.split(SC))) for idx in indices if idx]))
    try:
        dataset.dataset.fill_labels(preds, saved_path='pred_%s.csv' % resf_prefix, index=indices)
    except Exception as e:
        print(e)


def _prsn_cor(trues, preds):
    return np.corrcoef(trues, preds)[0, 1]


def save_model(model, optimizer, fpath='checkpoint.pth', in_wrapper=False, devq=None, **kwargs):
    print('Saving trained model...')
    if in_wrapper: model = model.module
    model = model.cpu() if devq and len(devq) > 0 else model
    checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    checkpoint.update(kwargs)
    torch.save(checkpoint, fpath)


def load_model(mdl_path):
    print('Loading previously trained model...')
    checkpoint = torch.load(mdl_path, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def _handle_model(model, dev_id=None, distrb=False):
    if (distrb):
        if (type(dev_id) is list):
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.cuda(dev_id)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id])
            raise NotImplementedError
    elif (dev_id is not None):
        if (type(dev_id) is list):
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.cuda(dev_id)
    return model


def main():
    if any(opts.task == t for t in ['bc5cdr-chem', 'bc5cdr-dz', 'shareclefe', 'ddi', 'chemprot', 'i2b2', 'hoc', 'mednli', 'biosses', 'clnclsts']):
        main_func = classify
    else:
        return
    if (opts.distrb):
        if (opts.np > 1): # Multi-process multiple GPU
            import torch.multiprocessing as mp
            mp.spawn(main_func, nprocs=len(opts.devq))
        else: # Single-process multiple GPU
            main_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    elif (opts.devq): # Single-process
        main_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    else:
        main_func(None) # CPU


if __name__ == '__main__':
    # Logging setting
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Parse commandline arguments
    op = OptionParser()
    op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
    op.add_option('-p', '--pid', default=0, action='store', type='int', dest='pid', help='indicate the process ID')
    op.add_option('-n', '--np', default=1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
    op.add_option('-f', '--fmt', default='npz', help='data stored format: csv, npz, or h5 [default: %default]')
    op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
    op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
    op.add_option('-j', '--epochs', default=1, action='store', type='int', dest='epochs', help='indicate the epoch used in deep learning')
    op.add_option('-z', '--bsize', default=64, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
    op.add_option('-o', '--omp', action='store_true', dest='omp', default=False, help='use openmp multi-thread')
    op.add_option('-g', '--gpunum', default=1, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
    op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    op.add_option('--gpumem', default=0.5, action='store', type='float', dest='gpumem', help='indicate the per process gpu memory fraction')
    op.add_option('--crsdev', action='store_true', dest='crsdev', default=False, help='whether to use heterogeneous devices')
    op.add_option('--distrb', action='store_true', dest='distrb', default=False, help='whether to distribute data over multiple devices')
    op.add_option('--distbknd', default='nccl', action='store', dest='distbknd', help='distribute framework backend')
    op.add_option('--disturl', default='env://', action='store', dest='disturl', help='distribute framework url')
    op.add_option('--earlystop', default=False, action='store_true', dest='earlystop', help='whether to use early stopping')
    op.add_option('--es_patience', default=5, action='store', type='int', dest='es_patience', help='indicate the tolerance time for training metric violation')
    op.add_option('--es_delta', default=float(5e-3), action='store', type='float', dest='es_delta', help='indicate the minimum delta of early stopping')
    op.add_option('--options_path', dest='options_path', help='ELMo option file')
    op.add_option('--weights_path', dest='weights_path', help='ELMo weight file')
    op.add_option('--maxlen', default=128, action='store', type='int', dest='maxlen', help='indicate the maximum sequence length for each samples')
    op.add_option('--maxtrial', default=50, action='store', type='int', dest='maxtrial', help='maximum time to try')
    op.add_option('--initln', default=False, action='store_true', dest='initln', help='whether to initialize the linear layer')
    op.add_option('--initln_mean', default=0., action='store', type='float', dest='initln_mean', help='indicate the mean of the parameters in linear model when Initializing')
    op.add_option('--initln_std', default=0.02, action='store', type='float', dest='initln_std', help='indicate the standard deviation of the parameters in linear model when Initializing')
    op.add_option('--weight_class', default=False, action='store_true', dest='weight_class', help='whether to drop the last incompleted batch')
    op.add_option('--droplast', default=False, action='store_true', dest='droplast', help='whether to drop the last incompleted batch')
    op.add_option('--do_norm', default=False, action='store_true', dest='do_norm', help='whether to do normalization')
    op.add_option('--norm_type', default='batch', action='store', dest='norm_type', help='normalization layer class')
    op.add_option('--do_lastdrop', default=False, action='store_true', dest='do_lastdrop', help='whether to apply dropout to the last layer')
    op.add_option('--lm_loss', default=False, action='store_true', dest='lm_loss', help='whether to apply dropout to the last layer')
    op.add_option('--do_crf', default=False, action='store_true', dest='do_crf', help='whether to apply CRF layer')
    op.add_option('--fchdim', default=0, action='store', type='int', dest='fchdim', help='indicate the dimensions of the hidden layers in the Embedding-based classifier, 0 means using only one linear layer')
    op.add_option('--pool', dest='pool', help='indicate the pooling strategy when selecting features: max or avg')
    op.add_option('--seq2seq', dest='seq2seq', help='indicate the seq2seq strategy when converting sequences of embeddings into a vector')
    op.add_option('--seq2vec', dest='seq2vec', help='indicate the seq2vec strategy when converting sequences of embeddings into a vector: pytorch-lstm, cnn, or cnn_highway')
    op.add_option('--ssfunc', dest='sentsim_func', help='indicate the sentence similarity metric')
    op.add_option('--lr', default=float(1e-3), action='store', type='float', dest='lr', help='indicate the learning rate of the optimizer')
    op.add_option('--wdecay', default=float(1e-5), action='store', type='float', dest='wdecay', help='indicate the weight decay of the optimizer')
    op.add_option('--lmcoef', default=0.5, action='store', type='float', dest='lmcoef', help='indicate the coefficient of the language model loss when fine tuning')
    op.add_option('--pdrop', default=0.2, action='store', type='float', dest='pdrop', help='indicate the dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler')
    op.add_option('--pthrshld', default=0.5, action='store', type='float', dest='pthrshld', help='indicate the threshold for predictive probabilitiy')
    op.add_option('--clipmaxn', default=0.25, action='store', type='float', dest='clipmaxn', help='indicate the max norm of the gradients')
    op.add_option('--resume', action='store', dest='resume', help='resume training model file')
    op.add_option('-i', '--input', help='input dataset')
    op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
    op.add_option('-u', '--task', default='ddi', type='str', dest='task', help='the task name [default: %default]')
    op.add_option('-m', '--model', default='elmo', type='str', dest='model', help='the model to be validated')
    op.add_option('--pretrained', dest='pretrained', help='pretrained model file')
    op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')
    op.add_option('--data_dir', dest='data_dir', help='indicate the data path')

    (opts, args) = op.parse_args()
    if len(args) > 0:
    	op.print_help()
    	op.error('Please input options instead of arguments.')
    	sys.exit(1)

    if (opts.gpuq is not None and not opts.gpuq.strip().isspace()):
    	opts.gpuq = list(range(torch.cuda.device_count())) if (opts.gpuq == 'auto' or opts.gpuq == 'all') else [int(x) for x in opts.gpuq.split(',') if x]
    elif (opts.gpunum > 0):
        opts.gpuq = list(range(opts.gpunum))
    else:
        opts.gpuq = []
    if (opts.gpuq and opts.gpunum > 0):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opts.gpuq[:opts.gpunum]))
        setattr(opts, 'devq', list(range(torch.cuda.device_count())))
    else:
        setattr(opts, 'devq', None)

    if (opts.options_path): LM_PARAMS['ELMo']['options_path'] = opts.options_path
    if (opts.weights_path): LM_PARAMS['ELMo']['weights_path'] = opts.weights_path
    if (opts.data_dir): DATA_PATH = opts.data_dir

    main()
