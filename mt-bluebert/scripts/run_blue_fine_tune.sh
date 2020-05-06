#!/bin/sh

if [ $# -ne 3 ]; then
  echo "fine_tune.sh <batch_size> <gpu> <task>"
  exit 1
fi

BATCH_SIZE=$1
gpu=$2
task=$3

echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

MODEL_NAME="bluebert-mt-dnn4-biomedical-pubmed_adam_answer_opt0_gc1_ggc1_2020-02-16T1213_97_stripped"
ROOT="bionlp2020"
BERT_PATH="$ROOT/bluebert_models/mt_dnn_bluebert_base_cased/$MODEL_NAME.pt"
DATA_DIR="$ROOT/blue_data/canonical_data/bert_uncased_lower"

if [ "$task" = "clinicalsts" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=30
  lr="5e-5"
elif [ "$task" = "biosses" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=20
  lr="5e-6"
elif [ "$task" = "mednli" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=10
  lr="1e-5"
elif [ "$task" = "mednli" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=10
  lr="5e-5"
elif [ "$task" = "i2b2-2010-re" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=1
  global_grad_clipping=1
  epochs=10
  lr="2e-5"
elif [ "$task" = "chemprot" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=20
  lr="2e-5"
elif [ "$task" = "ddi2013-type" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=10
  lr="5e-5"
elif [ "$task" = "bc5cdr-chemical" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=20
  lr="5e-5"
elif [ "$task" = "bc5cdr-disease" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=20
  lr="6e-5"
elif [ "$task" = "shareclefe" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=20
  lr="5e-5"
else
  echo "Cannot recognize $task"
  exit
fi

train_datasets=$task
test_datasets=$task

model_dir="$ROOT/checkpoints_blue/${MODEL_NAME}_${task}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python experiments/blue/blue_train.py \
  --data_dir ${DATA_DIR} \
  --init_checkpoint ${BERT_PATH} \
  --task_def experiments/blue/blue_task_def.yml \
  --batch_size "${BATCH_SIZE}" \
  --epochs ${epochs} \
  --output_dir "${model_dir}" \
  --log_file "${log_file}" \
  --answer_opt ${answer_opt} \
  --optimizer ${optim} \
  --train_datasets "${train_datasets}" \
  --test_datasets "${test_datasets}" \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --max_seq_len 128 \
  --not_save
