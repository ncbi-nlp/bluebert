"""
Usage:
    my_program.py [options]

Options:
    --do_train              Whether to run training.
    --do_test               Whether to run testing.
    --do_predict            Whether to run predicting.
    --do_debug

    --log_file=<file>           path for log file. [default: log.txt]
    --csv_logger=<file>         path for training log file. [default: log.csv]
    --output_dir=<dir>          The output directory where the model checkpoints will be written. [default: checkpoint]
    --dataset=<file>            dataset to use [default: dataset.csv]
    --train_dataset=<str>       [default: 1,2,3,4,5,6,7]
    --val_dataset=<str>         [default: 8]
    --test_dataset=<str>        [default: 9,10]
    --pred_dataset=<file        dataset to predict [default: unlabelled_dataset.csv]
    --test_predictions=<str>    [default: test_predictions.csv]
    --pred_predictions=<str>    [default: pred_predictions.csv]
    --best_model=<str>          best trained model. [default: best_model.h5]

    --bert_config_file=<str>    The config json file corresponding to the pre-trained BERT model.
    --init_checkpoint=<str>     Initial checkpoint (usually from a pre-trained BERT model).
    --vocab_file=<str>          The vocabulary file that the BERT model was trained on.
    --do_lower_case             Whether to lower case the input text.
    --epochs=<int>              Total number of training epochs to perform. [default: 3]
    --batch_size=<int>          Total batch size. [default: 8]
    --learning_rate=<float>     The initial learning rate for Adam. [default: 5e-5]
    --workers=<int>             number of workers for data processing [default: 3]
    --earlystop=<int>           [default: 5]
    --seed=<int>                random seed to use. [default: 123]
    --warmup_proportion=<float>     Proportion of training to perform linear learning rate warmup for. [default: 0.1]
    --max_seq_length=<int>      The maximum total input sequence length after WordPiece tokenization. [default: 128]

    --text_col=<str>            [default: x]
    --label_col=<str>           [default: y]
    --fold_col=<str>            [default: fold]

    --verbose

"""
import argparse
import json
import logging as log
import math
import os
import sys
from typing import List

import docopt
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras import callbacks
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Dropout, Masking, GlobalAveragePooling1D, Lambda
from keras.models import Model, load_model
from keras.utils import Sequence, to_categorical
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup, calc_train_steps, load_vocabulary, Tokenizer, \
    get_custom_objects
from tabulate import tabulate

from keras_image_app import pmetrics
from keras_image_app.image_utils import set_logger, load_instances, dump_model, dict_to_namespace, print_model, pick_device, \
    get_class_weights


class TextDataFrameIterator(Sequence):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer,
                 classes: List[str] = None,
                 x_col="text",
                 y_col="class",
                 seq_len=128,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 do_lower_case=True):
        self.dataframe = dataframe  # type: pd.DataFrame
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.x_col = x_col
        self.y_col = y_col
        self.seq_len = seq_len
        self.do_lower_case = do_lower_case
        self.tokenizer = tokenizer
        self.seed = seed

        if classes is None:
            self.classes = list(sorted(set(self.dataframe[self.y_col])))
        else:
            self.classes = classes
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def __getitem__(self, idx):
        tokens, labels = [], []
        batch = self.dataframe.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        for text, label in zip(batch[self.x_col], batch[self.y_col]):
            if self.do_lower_case:
                text = text.lower()
            token, _ = self.tokenizer.encode(text, max_len=self.seq_len)
            tokens.append(token)
            labels.append(self.class_indices[label])
        tokens = np.array(tokens)
        labels = to_categorical(labels, len(self.classes)).astype(int)
        return [tokens, np.zeros_like(tokens)], np.array(labels)

    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)


def get_args() -> argparse.Namespace:
    args = docopt.docopt(__doc__)
    namespace = dict_to_namespace(args, donot_convert={'val_dataset'})
    return namespace


def get_model(args):
    with open(args.bert_config_file, 'r') as fp:
        config = json.load(fp)

    bert_model = load_trained_model_from_checkpoint(
        config_file=args.bert_config_file,
        checkpoint_file=args.init_checkpoint,
        training=False,
        trainable=True,
        output_layer_num=1,
        seq_len=args.max_seq_length
    )
    bert_output = bert_model.outputs[0]
    # first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
    # x = Masking(mask_value=0.)(bert_output)
    # x = GlobalAveragePooling1D()(x)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. We assume that this has been pre-trained
    x = Lambda(lambda x: K.squeeze(x[:, 0:1, :], axis=1))(bert_output)
    x = Dense(config['hidden_size'], activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0., stddev=config['initializer_range']))(x)
    x = Dropout(0.1)(x)

    predictions = Dense(args.n_classes, activation='softmax',
                        kernel_initializer=TruncatedNormal(mean=0., stddev=0.02))(x)
    final_model = Model(inputs=bert_model.inputs, outputs=predictions)
    print_model(final_model)
    dump_model(os.path.join(args.output_dir, 'bert_structure.json'), final_model)
    return final_model


def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.random.seed(args.seed)

    if args.verbose:
        log.basicConfig(level=log.DEBUG, stream=sys.stdout)
    else:
        log.basicConfig(level=log.INFO, stream=sys.stdout)

    log.info('\n' + tabulate(sorted(vars(args).items())))
    set_logger(os.path.join(args.output_dir, args.log_file))

    pick_device()
    data = load_instances(args.dataset, args.label_col)
    classes = list(sorted(set(data[args.label_col])))
    args.n_classes = len(classes)

    token_dict = load_vocabulary(args.vocab_file)
    tokenizer = Tokenizer(token_dict)

    if args.do_train:
        folds = [i for i in args.train_dataset.split(',')]
        train_df = data[data['fold'].isin(folds)].reset_index(drop=True)
        train_generator = TextDataFrameIterator(
            dataframe=train_df,
            tokenizer=tokenizer,
            classes=classes,
            x_col=args.text_col,
            y_col=args.label_col,
            batch_size=args.batch_size,
            shuffle=True,
            seq_len=args.max_seq_length,
            seed=args.seed,
            do_lower_case=args.do_lower_case
        )

        folds = [i for i in args.val_dataset.split(',')]
        val_df = data[data['fold'].isin(folds)].reset_index(drop=True)
        val_generator = TextDataFrameIterator(
            dataframe=val_df,
            tokenizer=tokenizer,
            classes=classes,
            x_col=args.text_col,
            y_col=args.label_col,
            batch_size=args.batch_size,
            shuffle=False,
            seq_len=args.max_seq_length,
            do_lower_case=args.do_lower_case
        )

        total_steps, warmup_steps = calc_train_steps(
            num_example=len(train_df),
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_proportion=args.warmup_proportion,
        )

        model = get_model(args)
        earlystop = callbacks.EarlyStopping(
            monitor='val_loss', min_delta=K.epsilon(), patience=args.earlystop,
            verbose=1, mode='auto')
        best_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, args.best_model),
            save_best_only=True, save_weights_only=False,
            monitor='val_loss', mode='min', verbose=1)
        csv_logger = callbacks.CSVLogger(os.path.join(args.output_dir, args.csv_logger))

        callbacks_list = [earlystop, best_checkpoint, csv_logger]
        optimizer = AdamWarmup(
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            min_lr=1e-5,
            weight_decay=0.01,
            weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo']
        )
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        cw = get_class_weights(data, args.label_col, train_generator.class_indices)
        model.fit_generator(
            train_generator,
            class_weight=cw,
            use_multiprocessing=False,
            workers=args.workers,
            callbacks=callbacks_list,
            epochs=args.epochs,
            validation_data=val_generator,
            verbose=1)

    if args.do_test:
        folds = [i for i in args.test_dataset.split(',')]
        test_df = data[data['fold'].isin(folds)].reset_index(drop=True)
        test_generator = TextDataFrameIterator(
            dataframe=test_df,
            tokenizer=tokenizer,
            classes=classes,
            x_col=args.text_col,
            y_col=args.label_col,
            batch_size=args.batch_size,
            shuffle=False,
            seq_len=args.max_seq_length,
            do_lower_case=args.do_lower_case
        )

        print('Load from %s', os.path.join(args.output_dir, args.best_model))
        model = load_model(os.path.join(args.output_dir, args.best_model), custom_objects=get_custom_objects())
        # model.summary()
        y_score = model.predict_generator(
            test_generator,
            use_multiprocessing=False,
            workers=args.workers,
            verbose=1)

        y_pred = np.argmax(y_score, axis=1)

        pred_df = pd.DataFrame(y_score, columns=classes)
        pred_df = pred_df.assign(predictions=[classes[lbl] for lbl in y_pred])

        y_true = test_df.loc[:, args.label_col].values
        y_pred = pred_df['predictions'].values
        report = pmetrics.classification_report(y_true, y_pred, classes=classes)
        print(report.summary())
        # print('auc', pmetrics.auc(y_true, y_score, y_column=1)[0])

        result = pd.concat([test_df, pred_df], axis=1)
        result.to_csv(os.path.join(args.output_dir, args.test_predictions), index=False)

    if args.do_predict:
        test_df = load_instances(args.pred_dataset, args.label_col)
        test_generator = TextDataFrameIterator(
            dataframe=test_df,
            tokenizer=tokenizer,
            classes=None,
            x_col=args.text_col,
            y_col=args.label_col,
            batch_size=args.batch_size,
            shuffle=False,
            seq_len=args.max_seq_length,
            do_lower_case=args.do_lower_case
        )

        print('Load from %s', os.path.join(args.output_dir, args.best_model))
        model = load_model(os.path.join(args.output_dir, args.best_model), custom_objects=get_custom_objects())
        # model.summary()
        y_score = model.predict_generator(
            test_generator,
            use_multiprocessing=False,
            workers=args.workers,
            verbose=1)
        y_pred = np.argmax(y_score, axis=1)

        pred_df = pd.DataFrame(y_score, columns=classes)
        pred_df = pred_df.assign(predictions=[classes[lbl] for lbl in y_pred])
        result = pd.concat([test_df, pred_df], axis=1)
        result.to_csv(os.path.join(args.output_dir, args.pred_predictions), index=False)

    if args.do_debug:
        for dataset in [args.train_dataset, args.val_dataset, args.test_dataset]:
            folds = [i for i in dataset.split(',')]
            print('folds:', folds)
            sub_df = data[data['fold'].isin(folds)]
            generator = TextDataFrameIterator(
                dataframe=sub_df,
                tokenizer=tokenizer,
                x_col=args.text_col,
                y_col=args.label_col,
                batch_size=args.batch_size,
                shuffle=False,
                seq_len=args.max_seq_length,
            )
            for i, ([tokens, _], labels) in enumerate(generator):
                print(tokens.shape, type(tokens), labels.shape, type(labels))
                if i == 2:
                    break


if __name__ == '__main__':
    main()
