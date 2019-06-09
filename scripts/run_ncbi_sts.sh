#!/usr/bin/env bash

pubmed_uncased_L-12_H-768_A-12
pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12

##############################################################################
# Ours base
##############################################################################
CUDA_VISIBLE_DEVICES=2 python bert_ncbi/run_ncbi_sts.py \
  --task_name='clinicalsts' \
  --do_train=true \
  --do_eval=false \
  --do_test=true \
  --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --num_train_epochs=30.0 \
  --do_lower_case=true \
  --data_dir=/home/pengy6/data/bionlp2019/data/clinicalSTS \
  --output_dir=/home/pengy6/data/bionlp2019/data/outputs/clinicalSTS-bert-base

##############################################################################
# biobert
##############################################################################
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi_sts.py \
    --do_train=true \
    --do_eval=false \
    --do_test=true \
    --task_name="clinicalsts" \
    --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/vocab.txt \
    --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/bert_config.json \
    --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/biobert_model.ckpt \
    --max_seq_length=128 \
    --num_train_epochs=30.0 \
    --do_lower_case=False \
    --data_dir=/home/pengy6/data/sentence_similarity/data/BIOSSES-Dataset \
    --output_dir=/home/pengy6/data/sentence_similarity/data/outputs/BIOSSES-Dataset-biobert