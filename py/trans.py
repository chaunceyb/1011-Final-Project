import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import numpy as np
from collections import Counter
import time
# import datetime as dt
from tqdm import tqdm
# from masked_nll_loss import masked_nll_loss
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
# import sys
import os
import pdb
import torch.utils.data as Data
from torch.utils.data import Dataset
import argparse
import translate
import pickle as pkl
import pathlib


# CONSTANT DEFINATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#    torch.cuda.set_device(2)

BATCH_SIZE = 32
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

cons = [device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX]




# ---------------------------------------
# ---------------------------------------
# ---------------------------------------

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['train', 'eval', 'plot'])
parser.add_argument('-l', '--language', choices=['zh', 'vi'], default='zh',
                    help = '''Choose the language dataset to use.
                    Choices: zh, vi.''')
parser.add_argument('-m', '--mini', action='store_true', default=False,
                    help = 'Use mini dataset to debug.')
parser.add_argument('-e', '--example', action='store_true', default=True,
                    help = 'Print example translation while training.')
parser.add_argument('-r', '--readonly', action = 'store_true', default=False,
                    help = 'Do not write model and state into files. ')
parser.add_argument('-d', '--model', choices = ['attn', 'noattn', 'selfenc'], default='attn',
                    help = 'Choose the model type.')
parser.add_argument('-n', '--rnn', choices = ['gru', 'lstm'], default = 'gru',
                    help = 'Choose the type of RNN to use.')
parser.add_argument('--output', action = 'store_true', default = False,
                    help = 'Output the results of translation.')
parser.add_argument('--evalbeam', type=int, default=5,
                    help = 'The beam size used in the evaluation.')
parser.add_argument('--beamgroup', action = 'store_true', default = False,
                    help = 'Calculate the beam by the length of sentence.')
args = parser.parse_args()


MAX_LENGTH = 100

if args.language == 'zh':
    tgt_voc_size = 25000
    src_voc_size = 25000
elif args.language == 'vi':
    tgt_voc_size = 20000
    src_voc_size = 20000

src_lang, tgt_lang, train_loader, val_loader, test_loader = translate.load.load_data(args, cons, tgt_voc_size, src_voc_size)


# parameters

hidden_size = 256
n_layers = 2
# learning_rate=0.00001
learning_rate=0.0001

if args.model == 'attn':
    encoder = translate.encoder.EncoderRNN(args, cons, src_lang.vocab_size, hidden_size, n_layers).to(device)
    decoder = translate.decoder.AttnDecoderRNN(args, cons, 'general', hidden_size, tgt_lang.vocab_size, n_layers).to(device)
    log = translate.model_state.model_state()
elif args.model == 'noattn':
    encoder = translate.encoder.EncoderRNN(args, cons, src_lang.vocab_size, hidden_size, n_layers).to(device)
    decoder = translate.decoder.AttnDecoderRNN(args, cons, 'none', hidden_size, tgt_lang.vocab_size, n_layers).to(device)
    log = translate.model_state.model_state()
elif args.model == 'selfenc':
    encoder = translate.self_encoder.SelfEncoder(3, hidden_size,  hidden_size, 8, dropout=0.2, vocab_size=src_lang.vocab_size).to(device)
    decoder = translate.decoder.AttnDecoderRNN(args, cons, 'none', hidden_size, tgt_lang.vocab_size, n_layers).to(
        device)
    log = translate.model_state.model_state()


encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr= 3 * learning_rate)


try:
    spec = (args.language, args.model, args.rnn)
    pathlib.Path('../state/{}_{}_{}'.format(*spec)).mkdir(exist_ok=True)
    encoder.load_state_dict(torch.load('../state/{}_{}_{}/encoder_state.pkl'.format(*spec), map_location = device))
    decoder.load_state_dict(torch.load('../state/{}_{}_{}/decoder_state.pkl'.format(*spec), map_location = device))
    encoder_optimizer.load_state_dict(torch.load('../state/{}_{}_{}/encoder_optimizer_state.pkl'.format(*spec), map_location=device))
    decoder_optimizer.load_state_dict(torch.load('../state/{}_{}_{}/decoder_optimizer_state.pkl'.format(*spec), map_location=device))
    log = pkl.load(open('../state/{}_{}_{}/log.pkl'.format(*spec), 'rb'))
except:
    print('Failed to load model state.')

if args.action == 'train':
    translate.do.do_train(args, cons, train_loader, val_loader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                          log, src_lang, tgt_lang, n_epochs=50)
elif args.action == 'eval':
    translate.do.eval(args, cons, test_loader, encoder, decoder, src_lang, tgt_lang, group_by_len = args.beamgroup)

elif args.action == 'plot':
    translate.do.val_plot_align(args, cons, val_loader, encoder, decoder, src_lang, tgt_lang)

