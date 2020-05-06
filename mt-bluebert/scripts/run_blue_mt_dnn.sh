#!/bin/sh

if [ $# -ne 2 ]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi

prefix="blue-mt-dnn"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="clinicalsts,mednli,i2b2-2010-re,chemprot,ddi2013-type,bc5cdr-chemical,bc5cdr-disease,shareclefe"
test_datasets="clinicalsts,mednli,i2b2-2010-re,chemprot,ddi2013-type,bc5cdr-chemical,bc5cdr-disease,shareclefe"

ROOT="bionlp2020"
BERT_PATH="$ROOT/bluebert_models/bluebert_base/bluebert_base.pt"
DATA_DIR="$ROOT/blue_data/canonical_data/bert_uncased_lower"

answer_opt=0
optim="adam"
grad_clipping=1
global_grad_clipping=1
lr="5e-5"
epochs=100

model_dir="$ROOT/checkpoints_blue/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python experiments/blue/blue_train.py \
  --data_dir ${DATA_DIR} \
  --init_checkpoint ${BERT_PATH} \
  --task_def experiments/blue/blue_task_def.yml \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${model_dir}" \
  --log_file "${log_file}" \
  --answer_opt ${answer_opt} \
  --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --multi_gpu_on \
  --epochs ${epochs} \
  --max_seq_len 128
#  --model_ckpt ${MODEL_CKPT} \
#  --resume
