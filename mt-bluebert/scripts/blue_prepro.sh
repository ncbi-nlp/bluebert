#! /bin/sh

ROOT="blue_data"
BERT_PATH="${ROOT}/bluebert_models/bert_uncased_lower"
datasets="clinicalsts,biosses,mednli,i2b2-2010-re,chemprot,ddi2013-type,bc5cdr-chemical,bc5cdr-disease,shareclefe"

python experiments/blue/blue_prepro.py \
  --root_dir $ROOT \
  --task_def experiments/blue/blue_task_def.yml \
  --datasets $datasets \
  --overwrite

python experiments/blue/blue_prepro_std.py \
  --vocab $BERT_PATH/vocab.txt \
  --root_dir $ROOT/canonical_data \
  --task_def experiments/blue/blue_task_def.yml \
  --do_lower_case \
  --max_seq_len 128 \
  --datasets $datasets \
  --overwrite
