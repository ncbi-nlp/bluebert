# ELMo Fine-tuning
The python script `elmoft.py` provides utility functions for fine-tuning the [ELMo model](https://allennlp.org/elmo). We used the model pre-trained on PubMed in our paper [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474).

## Pre-trained models and benchmark datasets
Please prepare the pre-trained ELMo model files `options.json` as well as `weights.hdf5` and indicate their locations in the parameters `--options_path` as well as`--weights_path` when running the script. The model pre-trained on PubMed can be downloaded from the [ELMo website](https://allennlp.org/elmo)

The benchmark datasets can be downloaded from [https://github.com/ncbi-nlp/BLUE_Benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark)

## Fine-tuning ELMo
We assume the ELMo model has been downloaded at `$ELMO_DIR`, and the dataset has been downloaded at `$DATASET_DIR`.

### Sentence similarity

```bash
python elmoft.py \
  --task 'clnclsts' \
  --seq2vec 'boe' \
  --options_path $ELMO_DIR/options.json \
  --weights_path $ELMO_DIR/weights.hdf5 \
  --maxlen 256 \
  --fchdim 500 \
  --lr 0.001 \
  --pdrop 0.5 \
  --do_norm --norm_type batch \
  --do_lastdrop \
  --initln \
  --earlystop \
  --epochs 20 \
  --bsize 64 \
  --data_dir=$DATASET_DIR
```

The task can be 

- `clnclsts`:  Mayo Clinics clinical sentence similarity task
- `biosses`:  Biomedical Summarization Track sentence similarity task


### Named Entity Recognition

```bash
python elmoft.py \
  --task 'bc5cdr-chem' \
  --seq2vec 'boe' \
  --options_path $ELMO_DIR/options.json \
  --weights_path $ELMO_DIR/weights.hdf5 \
  --maxlen 128\
  --fchdim 500 \
  --lr 0.001 \
  --pdrop 0.5 \
  --do_norm --norm_type batch \
  --do_lastdrop \
  --initln \
  --earlystop \
  --epochs 20 \
  --bsize 64 \
  --data_dir=$DATASET_DIR
```

The task can be 

- `bc5cdr-chem`: BC5CDR chemical or disease task
- `bc5cdr-dz`: BC5CDR disease task
- `shareclefe`: ShARe/CLEFE task

### Relation Extraction

```bash
python elmoft.py \
  --task 'ddi' \
  --seq2vec 'boe' \
  --options_path $ELMO_DIR/options.json \
  --weights_path $ELMO_DIR/weights.hdf5 \
  --maxlen 128 \
  --fchdim 500 \
  --lr 0.001 \
  --pdrop 0.5 \
  --do_norm --norm_type batch \
  --initln \
  --earlystop \
  --epochs 20 \
  --bsize 64 \
  --data_dir=$DATASET_DIR
```

The task name can be 

- `ddi`: DDI 2013 task
- `chemprot`: BC6 ChemProt task
- `i2b2`: I2B2 relation extraction task

### Document multilabel classification

```bash
python elmoft.py \
  --task 'hoc' \
  --seq2vec 'boe' \
  --options_path $ELMO_DIR/options.json \
  --weights_path $ELMO_DIR/weights.hdf5 \
  --maxlen 128 \
  --fchdim 500 \
  --lr 0.001 \
  --pdrop 0.5 \
  --do_norm --norm_type batch \
  --initln \
  --earlystop \
  --epochs 50 \
  --bsize 64 \
  --data_dir=$DATASET_DIR
```

### Inference task

```bash
python elmoft.py \
  --task 'mednli' \
  --seq2vec 'boe' \
  --options_path $ELMO_DIR/options.json \
  --weights_path $ELMO_DIR/weights.hdf5 \
  --maxlen 128 \
  --fchdim 500 \
  --lr 0.0005 \
  --pdrop 0.5 \
  --do_norm --norm_type batch \
  --initln \
  --earlystop \
  --epochs 20 \
  --bsize 64 \
  --data_dir=$DATASET_DIR
```

## GPU Acceleration
If there is no GPU devices please indicate it using the parameter `-g 0` or `--gpunum 0`.  Otherwise, please indicate the index (starting from 0) of the GPU you want to use by setting the parameter `-q 0` or `--gpuq 0`.
