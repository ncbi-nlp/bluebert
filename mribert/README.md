## Automatic recognition of abdominal lymph nodes from clinical text

This repository provides codes and models of the BERT model for lymph node detection from MRI reports. 

## Pre-trained models

The pre-trained model weights, vocab, and config files can be downloaded from:

* [mribert](https://github.com/ncbi-nlp/bluebert/releases/tag/lymphnode)

## Fine-tuning BERT

We assume the MriBERT model has been downloaded at `$MriBERT_DIR`.

```bash
tstr=$(date +"%FT%H%M%S%N")
text_col="text"
label_col="label"
batch_size=32
train_dataset="train,dev"
val_dataset="dev"
test_dataset="test"
epochs=10

bert_dir=$MriBERT_DIR
dataset="$MriBERT_DIR/total_data.csv"
model_dir="MriBERT_DIR/mri_${tstr}"
test_predictions="predictions_mribert.csv"

# predict new
pred_dataset="$MriBERT_DIR/new_data.csv"
pred_predictions="new_data_predictions.csv"

export PYTHONPATH=.;$PYTHONPATH
python sequence_classification.py \
  --do_train \
  --do_test \
  --dataset "${dataset}" \
  --output_dir "${model_dir}" \
  --vocab_file $bert_dir/vocab.txt \
  --bert_config_file $bert_dir/bert_config.json \
  --init_checkpoint $bert_dir/mribert_model.ckpt \
  --text_col "${text_col}" \
  --label_col "${label_col}" \
  --batch_size "${batch_size}" \
  --train_dataset "${train_dataset}" \
  --val_dataset "${val_dataset}" \
  --test_dataset "${test_dataset}" \
  --pred_dataset "${pred_dataset}" \
  --epochs ${epochs} \
  --test_predictions ${test_predictions} \
  --pred_predictions ${pred_predictions}
```


## Citing MriBert

Peng Y, Lee S, Elton D, Shen T, Tang YX, Chen Q, Wang S, Zhu Y, Summers RM, Lu Z.
Automatic recognition of abdominal lymph nodes from clinical text.
In Proceedings of the ClinicalNLP Workshop. 2020.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of
Medicine and Clinical Center. 
This work was supported by the National Library of Medicine of the National Institutes of Health under award number 4R00LM013001.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NLM/NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NLM/NCBI's disclaimer policy is available.
