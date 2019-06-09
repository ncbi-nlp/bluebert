#!/usr/bin/env bash

biobert_pubmed
pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12
pubmed_uncased_L-12_H-768_A-12
uncased_L-12_H-768_A-12

##############################################################################
# Biobert
##############################################################################
CUDA_VISIBLE_DEVICES=1 python bert_ncbi/run_ncbi_ner.py \
  --do_prepare=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="ner" \
  --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/vocab.txt \
  --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/bert_config.json \
  --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/biobert_model.ckpt \
  --num_train_epochs=30.0 \
  --do_lower_case=False \
  --data_dir=/home/pengy6/data/bionlp2019/data/BC5CDR/chem/ \
  --output_dir=/home/pengy6/data/bionlp2019/data/outputs/BC5CDR-chem-biobert2
  

##############################################################################
# Ours base
##############################################################################
CUDA_VISIBLE_DEVICES=1 python bert_ncbi/run_ncbi_ner.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --task_name="i2b2" \
  --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/pubmed_uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12/bert_model.ckpt \
  --num_train_epochs=30.0 \
  --do_lower_case=true \
  --learning_rate=5e-5 \
  --data_dir=/home/pengy6/data/bionlp2019/data/i2b2-2012/ \
  --output_dir=/home/pengy6/data/bionlp2019/data/outputs/i2b2-2012-our-base-mimic
