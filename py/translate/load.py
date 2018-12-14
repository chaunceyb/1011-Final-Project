# Language Model

import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import Counter

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class Lang:
    def __init__(self, name):
        self.name = name

    def build_vocab(self, corpus, voc_size=0.95):
        all_words = []
        for i in corpus:
            all_words += i
        if (voc_size > 1):
            max_vocab_size = voc_size
        else:
            max_vocab_size = round(len(set(all_words)) * voc_size)
        words_counter = Counter(all_words)
        vocab, count = zip(*words_counter.most_common(max_vocab_size))
        self.vocab_size = len(vocab) + 4
        id2token = list(vocab)
        self.id2token = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + id2token
        self.token2id = dict(zip(vocab, range(4, 4 + len(vocab))))
        self.token2id['<PAD>'] = PAD_IDX
        self.token2id['<UNK>'] = UNK_IDX
        self.token2id['<SOS>'] = SOS_IDX
        self.token2id['<EOS>'] = EOS_IDX

    def one_hot_trans(self, corpus):
        return list(map(self.word_map, corpus))

    def word_map(self, sentence):
        return [self.token2id[i] if i in self.token2id else UNK_IDX for i in sentence]

# Dataset

class MyDataset(Dataset):
    def __init__(self, in_lang, out_lang, out_lines):
        assert len(in_lang) == len(out_lang)
        self.length = len(in_lang)
        self.in_lang = in_lang
        self.out_lang = out_lang
        self.add_eos = [False] * self.length
        self.out_lines = out_lines # tgt lines without tokenization

    def __getitem__(self, index):
        dat_in = self.in_lang[index]
        dat_out = self.out_lang[index]
        if not self.add_eos[index]:
            dat_out.append(EOS_IDX)
            self.add_eos[index] = True
        dat_out_lines = self.out_lines[index]

        return {
            "in": dat_in,
            "out": dat_out,
            "in_len": len(dat_in),
            "out_len": len(dat_out),
            'out_lines': dat_out_lines
        }

    def __len__(self):
        return self.length


def vocab_collate_func(batch):
    sort_order = np.argsort([x['in_len'] for x in batch])[::-1]
    batch[:] = [batch[i] for i in sort_order if batch[i]['in_len'] > 0]

    def seq_pad(sent, leng, max_len):
        return np.pad(np.array(sent),
                      pad_width=(0, max_len - leng),
                      mode="constant", constant_values=PAD_IDX)

    in_len_list = [x['in_len'] for x in batch]
    out_len_list = [x['out_len'] for x in batch]

    in_max_len = max(in_len_list)
    out_max_len = max(out_len_list)
    in_list = [seq_pad(x['in'], x['in_len'], in_max_len) for x in batch]
    out_list = [seq_pad(x['out'], x['out_len'], out_max_len) for x in batch]
    in_mask = [seq_pad([1] * x['in_len'], x['in_len'], in_max_len) for x in batch]
    out_mask = [seq_pad([1] * x['out_len'], x['out_len'], out_max_len) for x in batch]
    out_org = [x['out_lines'] for x in batch]

    return {
               "sen": torch.from_numpy(np.array(in_list)).long().to(device),
               "sen_len": torch.LongTensor(in_len_list).to(device),
               'mask': torch.ByteTensor(in_mask).to(device),
           }, {
               "sen": torch.from_numpy(np.array(out_list)).long().to(device),
               "sen_len": torch.LongTensor(out_len_list).to(device),
               'mask': torch.ByteTensor(out_mask).to(device),
               'sen_org': out_org,
           }

def load_data(args, cons, tgt_voc_size = 15000, src_voc_size = 8000):
    global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons
    path = '../data/' + ('mini-' if args.mini else 'iwslt-') + args.language + '-en/'
    # Train token
    tgt_lines_train = open(path+'train.tok.en', encoding='utf-8').read().strip().split('\n')
    src_lines_train = open(path+'train.tok.' + args.language, encoding='utf-8').read().strip().split('\n')
    tgt_token_train = [i.split() for i in tgt_lines_train]
    src_token_train = [i.strip().replace('    ', ' . ').replace('   ', ' , ').split() for i in src_lines_train]
    # Val token
    tgt_lines_dev = open(path+'dev.tok.en', encoding='utf-8').read().strip().split('\n')
    src_lines_dev = open(path+'dev.tok.' + args.language, encoding='utf-8').read().strip().split('\n')
    tgt_token_dev = [i.split() for i in tgt_lines_dev]
    src_token_dev = [i.strip().replace('    ', ' . ').replace('   ', ' , ').split() for i in src_lines_dev]
    # test token
    tgt_lines_test = open(path+'test.tok.en', encoding='utf-8').read().strip().split('\n')
    src_lines_test = open(path+'test.tok.' + args.language, encoding='utf-8').read().strip().split('\n')
    tgt_token_test = [i.split() for i in tgt_lines_test]
    src_token_test = [i.strip().replace('    ', ' . ').replace('   ', ' , ').split() for i in src_lines_test]
    # Language vocab
    tgt_lang = Lang("en")
    tgt_lang.build_vocab(tgt_token_train, voc_size=tgt_voc_size)
    tgt_indice_train = tgt_lang.one_hot_trans(tgt_token_train)
    tgt_indice_dev = tgt_lang.one_hot_trans(tgt_token_dev)
    tgt_indice_test = tgt_lang.one_hot_trans(tgt_token_test)

    src_lang = Lang(args.language)
    src_lang.build_vocab(src_token_train, voc_size=src_voc_size)
    src_indice_train = src_lang.one_hot_trans(src_token_train)
    src_indice_dev = src_lang.one_hot_trans(src_token_dev)
    src_indice_test = src_lang.one_hot_trans(src_token_test)

    # build dataset
    train_dataset = MyDataset(src_indice_train, tgt_indice_train, tgt_lines_train)
    val_dataset = MyDataset(src_indice_dev, tgt_indice_dev, tgt_lines_dev)
    test_dataset = MyDataset(src_indice_test, tgt_indice_test, tgt_lines_test)
    # build loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=vocab_collate_func,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=vocab_collate_func,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=vocab_collate_func,
                                             shuffle=False)

    return src_lang, tgt_lang, train_loader, val_loader, test_loader

