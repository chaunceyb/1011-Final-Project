# 1011 Final Project

## Usage

Please run the `trans.py` in the `./py` directory

```
usage: trans.py [-h] [-l {zh,vi}] [-m] [-e] [-r] [-d {attn,noattn,selfenc}]
                [-n {gru,lstm}]
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
```

Before running the script, make sure:

-  the data is copied into the `./data/` directory. 
- the directory `./state/`, `./display` exist.

To train the self-attention model, please use the python file in the `./other/` directory. 



## Acknowlegment

