#!/usr/bin/env bash

/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/biobert_model.ckpt
/home/pengy6/data/sentence_similarity/bert-models/pubmed_NOTEEVENTS_uncased_L-12_H-768_A-12/model.ckpt-200000
/home/pengy6/data/sentence_similarity/bert-models/pubmed_uncased_L-12_H-768_A-12/model.ckpt-549000
/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_model.ckpt

##############################################################################
# biobert
##############################################################################
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi.py \
    --do_train=true \
    --do_eval=false \
    --do_predict=true \
    --task_name="i2b2" \
    --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/vocab.txt \
    --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/bert_config.json \
    --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/biobert_pubmed/biobert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=/home/pengy6/data/bionlp2019/data/i2b2-2010 \
    --output_dir=/home/pengy6/data/bionlp2019/data/outputs/i2b2-2010-biobert \
    --do_lower_case=False

##############################################################################
# Ours base
##############################################################################
CUDA_VISIBLE_DEVICES=2 python bert_ncbi/run_ncbi.py \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --task_name="mednli" \
    --vocab_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/home/pengy6/data/sentence_similarity/bert-models/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=/home/pengy6/data/bionlp2019/data/mednli \
    --output_dir=/home/pengy6/data/bionlp2019/data/outputs/mednli-bert-base \
    --do_lower_case=True

