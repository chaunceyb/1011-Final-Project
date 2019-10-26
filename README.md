# 💬Neural Machine Translation

![](https://img.shields.io/pypi/pyversions/torch)

> This is originally the final project of *DS*- *GA 1011*: Natural Language Processing with Representation Learning

## Usage

```
usage: trans.py [-h] [-l {zh,vi}] [-m] [-e] [-r] [-d {attn,noattn,selfenc}]
                [-n {gru,lstm}] [--output] [--evalbeam EVALBEAM] [--beamgroup]
                {train,eval,plot}

positional arguments:
  {train,eval,plot}

optional arguments:
  -h, --help            show this help message and exit
  -l {zh,vi}, --language {zh,vi}
                        Choose the language dataset to use. Choices: zh, vi.
  -m, --mini            Use mini dataset to debug.
  -e, --example         Print example translation while training.
  -r, --readonly        Do not write model and state into files.
  -d {attn,noattn,selfenc}, --model {attn,noattn,selfenc}
                        Choose the model type.
  -n {gru,lstm}, --rnn {gru,lstm}
                        Choose the type of RNN to use.
  --output              Output the results of translation.
  --evalbeam EVALBEAM   The beam size used in the evaluation.
  --beamgroup           Calculate the beam by the length of sentence.
```

## Getting Started

To train a translation model, below is a simple example.

First, make sure:

-  the data is copied into the `./data/` directory.
- the directory `./state/` exist to save model states.

Then, in the `./py/` directory, run the following command

```bash
python trans.py train
```

Then the script will start to train with default configurations (dataset: `iwslt-zh-en`, model: GRU with attention). The model parameters will be saved after every epoch with training loss and validation BLEU score, in the `./state` directory.

To evaluate the performance of the model on test set, run the follow command

```bash
python trans.py eval
```

## Detailed Usage

#### Selecting the language dataset

By adding the `-l vi` argument, the model will be trained using Vi-En corpus. Please make sure `iwslt-vi-en` is in the data folder.

#### Options for development and debugging

Since the corpus file is very large, you can prepare a smaller subset of the dataset during developing and debugging. Save the reduced dataset in the `./data/mini-{src_lang}-{tgt_lang}/` directory

Also, adding `-r` will enable the read only mode, preventing the script to overwrite existing saved model

#### Choosing the RNN

While we use GRU as our default RNN, you can choose `-n lstm` to switch to LSTM RNN.

#### Attention

Besides regular attention, you can also select model without attention, or self-attention as encoder. A fully self-attention model is available in `./other/` directory.

#### Plotting the Attention Alignment

Using the argument `plot` can create a graph of attention alignment in the `./display/` directory. Note that some characters may not render in all environments due to missing fonts. 

## Features for Evaluations

Below are the options for the evaluation. 

#### Save translations

Adding `--output` during evalutaion can save the translations with their source sentences and reference translation respectively into a txt file in `./display/`. Please make sure that the directory exists.

#### Use customize beam size

Using `--evalbeam 10` will change the beam size to 10 from 5, the default value.

#### Generate group beam score by length

Adding `--beamgroup` will generate beam score in several group of sentence lengths. The results will be saved as a dictionary in the `./display` directory

## Other

- Interrupt the training (with <kbd>Ctrl</kbd> + <kbd>c</kbd>) and resume training is supported. The current number of epochs will also be recorded so feel free to stop and resume. If wish to train from the beginning, simply delete the corresponding folding in `./state`
- Since we use the attention described by [Luong](https://arxiv.org/abs/1508.04025) and used teacher forcing, the decoder is implemented to input the whole sentence all at once during training. This will speed up the training time, but might cause limitations for future adaptations. 
- Running the training with a Tesla P100 GPU on Google Cloud, an epoch of training and validate will cost approximately 10 minutes for Zh-En task and 6 minutes for Vi-En task with GRU and attention.

## References

Part of the codes are adapted from existing, open-source projects. Includes:

- Our defination of the beam class is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) , which is distributed under MIT License. Some of our work is also inspired by this project, though implemented individually. 
- Our self-attention scripts are based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- The BLEU score calculation is from [SacreBLEU](https://github.com/mjpost/sacreBLEU). It is licensed under the Apache 2.0. The original license is in the `LICENSE` directory. 

