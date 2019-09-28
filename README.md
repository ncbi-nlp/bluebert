# NCBI BERT

**\*\*\*\*\* New July 11th, 2019: preprocessed PubMed texts \*\*\*\*\***

We uploaded the [preprocessed PubMed texts](https://github.com/ncbi-nlp/NCBI_BERT/blob/master/README.md#pubmed)  that were used to pre-train the NCBI_BERT models.

-----

This repository provides codes and models of NCBI BERT, pre-trained on PubMed abstracts and clinical notes ([MIMIC-III](https://mimic.physionet.org/)). Please refer to our paper [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474) for more details.

## Pre-trained models and benchmark datasets

The pre-trained NCBI BERT weights, vocab, and config files can be downloaded from: 

* [NCBI_BERT-Base, Uncased, PubMed](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12.zip): This model was pretrained on PubMed abstracts.
* [NCBI_BERT-Base, Uncased, PubMed+MIMIC-III](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip): This model was pretrained on PubMed abstracts and MIMIC-III.
* [NCBI_BERT-Large, Uncased, PubMed](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-24_H-1024_A-16.zip): This model was pretrained on PubMed abstracts.
* [NCBI_BERT-Large, Uncased, PubMed+MIMIC-III](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16.zip): This model was pretrained on PubMed abstracts and MIMIC-III.

The benchmark datasets can be downloaded from [https://github.com/ncbi-nlp/BLUE_Benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark)

## Fine-tuning NCBI BERT

We assume the NCBI BERT model has been downloaded at `$NCBI_BERT_DIR`, and the dataset has been downloaded at `$DATASET_DIR`.

Add local directory to `$PYTHONPATH` if needed.

```bash
export PYTHONPATH=.;$PYTHONPATH
```

### Sentence similarity

```bash
CUDA_VISIBLE_DEVICES=2 python bert_ncbi/run_ncbi_sts.py \
  --task_name='sts' \
  --do_train=true \
  --do_eval=false \
  --do_test=true \
  --vocab_file=$NCBI_BERT_DIR/vocab.txt \
  --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
  --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --num_train_epochs=30.0 \
  --do_lower_case=true \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR
```


### Named Entity Recognition

```bash
CUDA_VISIBLE_DEVICES=1 python bert_ncbi/run_ncbi_ner.py \
  --do_prepare=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="bc5cdr" \
  --vocab_file=$NCBI_BERT_DIR/vocab.txt \
  --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
  --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
  --num_train_epochs=30.0 \
  --do_lower_case=False \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR
```

The task name can be 

- `bc5cdr`: BC5CDR chemical or disease task
- `clefe`: ShARe/CLEFE task

### Relation Extraction

```bash
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi.py \
    --do_train=true \
    --do_eval=false \
    --do_predict=true \
    --task_name="chemprot" \
    --vocab_file=$NCBI_BERT_DIR/vocab.txt \
    --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
    --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_lower_case=False
```

The task name can be 

- `chemprot`: BC6 ChemProt task
- `ddi`: DDI 2013 task
- `i2b2_2010`: I2B2 2010 task

### Document multilabel classification

```bash
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi_multi_labels.py \
  --task_name="hoc" \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --vocab_file=$NCBI_BERT_DIR/vocab.txt \
  --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
  --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --num_classes=20 \
  --num_aspects=10 \
  --aspect_value_list="0,1" \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR
```

### Inference task

```bash
CUDA_VISIBLE_DEVICES=0 python bert_ncbi/run_ncbi.py \
    --do_train=true \
    --do_eval=false \
    --do_predict=true \
    --task_name="mednli" \
    --vocab_file=$NCBI_BERT_DIR/vocab.txt \
    --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
    --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_lower_case=False
```

## <a name="pubmed"></a>Preprocessed PubMed texts

We provide [preprocessed PubMed texts](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/pubmed_uncased_sentence_nltk.txt.tar.gz) that were used to pre-train the NCBI BERT models. The corpus contains ~4000M words extracted from the [PubMed ASCII code version](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/). Other operations include

*  lowercasing the text
*  removing speical chars `\x00`-`\x7F`
*  tokenizing the text using the [NLTK Treebank tokenizer](https://www.nltk.org/_modules/nltk/tokenize/treebank.html)

Below is a code snippet for more details.

```python
   value = value.lower()
   value = re.sub(r'[\r\n]+', ' ', value)
   value = re.sub(r'[^\x00-\x7F]+', ' ', value)

   tokenized = TreebankWordTokenizer().tokenize(value)
   sentence = ' '.join(tokenized)
   sentence = re.sub(r"\s's\b", "'s", sentence)
```

### Pre-training with BERT

Afterwards, we used the following code to generate pre-training data. Please see https://github.com/google-research/bert for more details.

```bash
python bert/create_pretraining_data.py \
  --input_file=pubmed_uncased_sentence_nltk.txt \
  --output_file=pubmed_uncased_sentence_nltk.tfrecord \
  --vocab_file=bert_uncased_L-12_H-768_A-12_vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

We used the following code to train the BERT model. Please do not include `init_checkpoint` if you are pre-training from scratch. Please see https://github.com/google-research/bert for more details.

```bash
python run_pretraining.py \
  --input_file=pubmed_uncased_sentence_nltk.tfrecord \
  --output_dir=$NCBI_BERT_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$NCBI_BERT_DIR/bert_config.json \
  --init_checkpoint=$NCBI_BERT_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

## Citing NCBI BERT

*  Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.

```
@InProceedings{peng2019transfer,
  author    = {Yifan Peng and Shankai Yan and Zhiyong Lu},
  title     = {Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets},
  booktitle = {Proceedings of the 2019 Workshop on Biomedical Natural Language Processing (BioNLP 2019)},
  year      = {2019},
}
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of
Medicine and Clinical Center. This work was supported by the National Library of Medicine of the National Institutes of Health under award number K99LM013001-01.

We are also grateful to the authors of BERT and ELMo to make the data and codes publicly available.

We would like to thank Dr Sun Kim for processing the PubMed texts.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NCBI's disclaimer policy is available.
