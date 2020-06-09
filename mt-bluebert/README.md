# Multi-Task Learning on BERT for Biomedical Text Mining

This repository provides codes and models of the Multi-Task Learning on BERT for Biomedical Text Mining. 
The package is based on [`mt-dnn`](https://github.com/namisan/mt-dnn).

## Pre-trained models

The pre-trained MT-BlueBERT weights, vocab, and config files can be downloaded from:

* [mt-bluebert-biomedical](https://github.com/yfpeng/mt-bluebert/releases/download/0.1/mt-bluebert-biomedical.pt)
* [mt-bluebert-clinical](https://github.com/yfpeng/mt-bluebert/releases/download/0.1/mt-bluebert-clinical.pt)

The benchmark datasets can be downloaded from [https://github.com/ncbi-nlp/BLUE_Benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark)

## Quick start

### Setup Environment
1. python3.6
2. install requirements
```bash
pip install -r requirements.txt
```

### Download data
Please refer to download BLUE_Benchmark: https://github.com/ncbi-nlp/BLUE_Benchmark


### Preprocess data
```bash
bash ncbi_scripts/blue_prepro.sh
```

### Train a MT-DNN model
```bash
bash ncbi_scripts/run_blue_mt_dnn.sh
```

### Fine-tune a model
```bash
bash ncbi_scripts/run_blue_fine_tune.sh
```

### Convert Tensorflow BERT model to the MT-DNN format
```bash
python ncbi_scripts/convert_tf_to_pt.py --tf_checkpoint_root $SRC_ROOT --pytorch_checkpoint_path $DEST --encoder_type 1```
```

## Citing MT-BLUE

Peng Y, Chen Q, Lu Z. An Empirical Study of Multi-Task Learning on BERT
for Biomedical Text Mining. In Proceedings of the 2020 Workshop on Biomedical
Natural Language Processing (BioNLP 2020). 2020.

```
@InProceedings{peng2019transfer,
  author    = {Yifan Peng and Qingyu Chen and Zhiyong Lu},
  title     = {An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining},
  booktitle = {Proceedings of the 2020 Workshop on Biomedical Natural Language Processing (BioNLP 2020)},
  year      = {2020},
}
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of
Medicine. This work was supported by the National Library of Medicine of the National Institutes of Health under award number K99LM013001-01.

We are also grateful to the authors of BERT and mt-dnn to make the data and codes publicly available. 

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NLM/NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NLM/NCBI's disclaimer policy is available.
