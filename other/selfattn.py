
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import string
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
import random
import logging
import os

from tensorboardX import SummaryWriter 
writer = SummaryWriter('runs/exp-1')


# In[2]:


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
info_handler = logging.FileHandler("selfattn.log")
info_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)


# In[4]:


PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class Lang:
    def __init__(self, name, is_out = False):
        self.name = name
        self.is_out = is_out
    
    def build_vocab(self, corpus, voc_size=0.90):
        all_words = []
        for i in corpus:
            all_words += i
        if(voc_size > 1):
            max_vocab_size = voc_size
        else:
            max_vocab_size = round(len(set(all_words)) * voc_size)
        words_counter = Counter(all_words)
        vocab, count = zip(*words_counter.most_common(max_vocab_size))
        self.vocab_size = len(vocab)+4
        id2token = list(vocab)
        self.id2token = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + id2token
        self.token2id = dict(zip(vocab, range(4,4+len(vocab))))
        self.token2id['<PAD>'] = PAD_IDX 
        self.token2id['<UNK>'] = UNK_IDX
        self.token2id['<SOS>'] = SOS_IDX 
        self.token2id['<EOS>'] = EOS_IDX
    
    def one_hot_trans(self, corpus):
        return list(map(self.word_map, corpus))
    
    def word_map(self, sentence):
        if self.is_out:
            return [self.token2id[i] if i in self.token2id else UNK_IDX for i in sentence] + [EOS_IDX]
        else:
            return [self.token2id[i] if i in self.token2id else UNK_IDX for i in sentence]


# In[4]:


import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
class MyDataset(Dataset):
    def __init__(self, in_lang, out_lang):
        assert len(in_lang) == len(out_lang)
        self.length = len(in_lang)
        self.in_lang = in_lang
        self.out_lang = out_lang
        
    def __getitem__(self, index):
        dat_in = self.in_lang[index]
        dat_out = self.out_lang[index]
        dat_out = np.concatenate([dat_out, [EOS_IDX]])
        
        return {
        "in": dat_in,
        "out": dat_out,
        "in_len": len(dat_in),
        "out_len": len(dat_out)}
    
    def __len__(self):
        return self.length


# In[9]:


def vocab_collate_func(batch):
    in_list = []
    out_list = []
    in_len_list = []
    out_len_list = []
    
    for datum in batch:
        if(datum["in_len"]>0 and datum["out_len"]>0):
            in_len_list.append(datum["in_len"])
            out_len_list.append(datum["out_len"])
        
    in_max_len = max(in_len_list)
    out_max_len = max(out_len_list)
    
    for datum in batch:
        if(datum["in_len"]>0 and datum["out_len"]>0):
            in_lang = np.pad(np.array(datum["in"]),
                        pad_width=((0, in_max_len-datum["in_len"])),
                        mode="constant", constant_values=PAD_IDX)
            in_list.append(in_lang)
            out_lang = np.pad(np.array(datum["out"]),
                          pad_width = ((0, out_max_len-datum["out_len"])),
                         mode="constant", constant_values=PAD_IDX)
            out_list.append(out_lang)
    
    out_order = np.argsort(out_len_list)[::-1]
    out_list = np.array(out_list)[out_order]
    out_len_list = np.array(out_len_list)[out_order]
    
    in_list = np.array(in_list)[out_order]
    in_len_list = np.array(in_len_list)[out_order]
    in_unsort_idx = np.zeros(len(in_list))
    
    for k in range(int(np.ceil(len(in_list)/BATCH_SIZE))):
        end = min((k+1)*BATCH_SIZE, len(in_list))
        sub = range(k*BATCH_SIZE, end)
        tmp = in_len_list[sub]
        in_order = np.argsort(tmp)[::-1]
        in_len_list[sub] = in_len_list[sub][in_order]
        in_list[sub] = in_list[sub][in_order]
        in_unsort_idx[sub] = np.argsort(in_order)
    
    return {
        "sen": torch.from_numpy(np.array(in_list)).long().to(device),
        "sen_len": torch.LongTensor(in_len_list).to(device),
        "sen_unsort_idx": torch.from_numpy(in_unsort_idx).long().to(device)
    },{
     "sen": torch.from_numpy(np.array(out_list)).long().to(device),
     "sen_len": torch.LongTensor(out_len_list).to(device),
    }


# # Encoder & Decoder

# ## EncoderDecoder

# In[10]:


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# ## Encoder

# In[11]:


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[12]:


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[13]:


class LayerNorm(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_embeddings, dropout):
        super(EncoderRNN, self).__init__()
        self.num_layers, self.hidden_size = num_layers, hidden_size

        self.embedding = nn.Embedding(num_embeddings, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout=0)
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        # use zero init for GRU layer0 bias
        self.gru.bias_ih_l0.data.zero_()
        self.gru.bias_hh_l0.data.zero_()
        self.context = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_trans = nn.Linear(self.hidden_size, self. hidden_size)
        

    
    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()

        return hidden
    
    def forward(self, x):
        batch_size, seq_len = x['sen'].size()
        self.hidden = self.init_hidden(batch_size).to(device)
        embed = self.embedding(x["sen"])
        batch_size, seq_len = x['sen'].size()
        self.hidden = self.init_hidden(batch_size).to(device)
        output = torch.nn.utils.rnn.pack_padded_sequence(embed, x['sen_len'], batch_first=True)
        _, context = self.gru(output, self.hidden)
        context = context.transpose(0,1)[x['sen_unsort_idx']].transpose(0,1)
        context = torch.tanh(self.context(context))
        self.hidden = torch.tanh(self.hidden_trans(context))
        return context, self.hidden


# In[14]:


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# In[15]:


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ## Decoder

# In[16]:


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In[17]:


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[18]:


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# In[19]:


class BianDecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, middle_size, num_embeddings, dropout):
        super(BianDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.middle_size = middle_size
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout=0)
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        self.gru.bias_ih_l0.data.zero_()
        self.gru.bias_hh_l0.data.zero_()
        
        # Define layers
        self.embedding = nn.Embedding(num_embeddings, emb_size)
        self.gru = nn.GRU(emb_size+self.hidden_size, self.hidden_size, num_layers, dropout=0)
        
        
        self.linear1 = nn.Linear(self.hidden_size, middle_size*2)
        self.linear2 = nn.Linear(middle_size, num_embeddings)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, word_input, last_hidden, context):
        batch_size = len(word_input)
        emb = self.embedding(word_input).unsqueeze(0)
        emb = self.dropout(emb)
        
        rnn_input = torch.cat((emb,context), dim=2)
        rnn_out, hidden = self.gru(rnn_input, last_hidden)
        
        out = rnn_out.squeeze(0)
        out = self.linear1(out)
        out = out.view(batch_size, self.middle_size, 2).max(2)[0]
        out = self.dropout(out)
        out = self.linear2(out)
        out = out - out.max(dim=1)[0].unsqueeze(1)
        out = F.log_softmax(out, dim=1)
        return out, hidden


# # Attention

# In[20]:


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))              / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# In[21]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# # Feed Forward

# In[22]:


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# # Full Model

# In[23]:


def make_model(src_vocab, tgt_vocab, dropout, N=6, 
               d_model=512, d_ff=2048, h=8):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


# # Training Model

# In[24]:


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask =                 self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# In[25]:


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# In[26]:



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# In[27]:


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# In[28]:


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm


# In[ ]:


V = 11
criterion = LabelSmoothing(size=V, padding_idx=PAD_IDX, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def run_epoch(data_iter, model, loss_compute):

    total_loss = 0
    total_tokens =0
    for in_sens, out_sens in data_iter:
        out = model.forward(in_sens["sen"][in_sens["sen_unsort_idx"]], out_sens["sen"], 
                            in_sens["sen_len"], out_sens["sen_len"])
        loss = loss_compute(out, out_sens["sen"], len(out_sens["sen_len"])
        total_loss += loss
        total_tokens += len(out_sens["sen_len"])
    return total_loss / total_tokens

                            
                            


# # Implementation

# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path =  '../data/iwslt-zh-en/'
en_token_train = open(path+'train.tok.en', encoding='utf-8').read().strip().split('\n')
zh_token_train = open(path+'train.tok.zh', encoding='utf-8').read().strip().split('\n')
en_token_train = [i.split() for i in en_token_train]
zh_token_train = [i.split() for i in zh_token_train]
en_token_dev = open(path+'dev.tok.en', encoding='utf-8').read().strip().split('\n')
zh_token_dev = open(path+'dev.tok.zh', encoding='utf-8').read().strip().split('\n')
en_token_dev = [i.split() for i in en_token_dev]
zh_token_dev = [i.split() for i in zh_token_dev]


# In[33]:


PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

en_lang = Lang("En", True)
en_lang.build_vocab(en_token_train, voc_size=50000)
en_indice_train = en_lang.one_hot_trans(en_token_train)
en_indice_dev = en_lang.one_hot_trans(en_token_dev)
zh_lang = Lang("Zh")
zh_lang.build_vocab(zh_token_train, voc_size=50000)
zh_indice_train = zh_lang.one_hot_trans(zh_token_train)
zh_indice_dev = zh_lang.one_hot_trans(zh_token_dev)

train_dataset = MyDataset(zh_indice_train, en_indice_train)
val_dataset = MyDataset(zh_indice_dev, en_indice_dev)


BATCH_SIZE = 64

train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE * 20,
                                collate_fn=vocab_collate_func,
                                shuffle=True)

val_ = torch.utils.data.DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE * 20,
                                collate_fn=vocab_collate_func,
                                shuffle=False)


# In[34]:


emb_size = 256
hidden_size = 1000
num_layers = 1
num_embeddings = zh_lang.vocab_size

encoder = EncoderRNN(emb_size, hidden_size, num_layers, num_embeddings, dropout=0.5).to(device)
decoder = BianDecoderRNN(emb_size, hidden_size, num_layers, 500, en_lang.vocab_size, dropout=0.5).to(device)

#encoder.load_state_dict(torch.load("encoder_state_noattn1.pkl"))
#decoder.load_state_dict(torch.load("decoder_state_noattn1.pkl"))


learning_rate = 0.0005

model = make_model(50000,50000, N=6)
model.cuda()
criterion = LabelSmoothing(size=50000, padding_idx=PAD_IDX, smoothing=0.1)
criterion.cuda()
model_par = nn.DataParallel(model, device_ids=devices)


# In[34]:


num_epochs = 100
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
for epoch in range(num_epochs):
    model_par.train()
    run_epoch(Batch(in_sens["sens"][in_sens["sen_unsort_idx"]],out_sens["sens"],PAD_IDX) for in_sens, out_sens in train_iter), 
              model_par, 
              MultiGPULossCompute(model.generator, criterion, 
                                  devices=devices, opt=model_opt))
    model_par.eval()
    loss = run_epoch(Batch(in_sens["sens"][in_sens["sen_unsort_idx"]],out_sens["sens"],PAD_IDX) for in_sens, out_sens in valid_iter), 
                      model_par, 
                      MultiGPULossCompute(model.generator, criterion, 
                      devices=devices, opt=None))
    print(loss)

