#!/usr/bin/env bash

biobert_pubmed
pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12
pubmed_uncased_L-12_H-768_A-12

##############################################################################
# biobert
##############################################################################
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi_multi_labels.py \
  --task_name="sentiment_analysis" \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/vocab.txt \
  --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/bert_config.json \
  --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/biobert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --num_classes=20 \
  --num_aspects=10 \
  --aspect_value_list="0,1" \
  --data_dir=/home/pengy6/data/sentence_similarity/data/hoc \
  --output_dir=/home/pengy6/data/sentence_similarity/data/outputs/hoc-biobert2


##############################################################################
# Ours base
##############################################################################
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi_multi_labels.py \
  --task_name="sentiment_analysis" \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --num_classes=20 \
  --num_aspects=10 \
  --aspect_value_list="0,1" \
  --data_dir=/home/pengy6/data/bionlp2019/data/hoc \
  --output_dir=/home/pengy6/data/bionlp2019/data/outputs/hoc-bert-base