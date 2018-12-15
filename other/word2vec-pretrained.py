import gensim

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

path = ''
token_train = open(path+'train.tok.language', encoding='utf-8').read().strip().split('\n')
token_train = [i.split() for i in token_train]

class Lang:
    def __init__(self, name):
        self.name = name
    
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
        
        pkl.dump(self.token2id, open( upload + "token2id.p", "wb" ))
        pkl.dump(self.id2token, open( upload + "id2token.p", "wb" ))
    
    def one_hot_trans(self, corpus):
        return list(map(self.word_map, corpus))
    
    def word_map(self, sentence):
        return [self.token2id[i] if i in self.token2id else UNK_IDX for i in sentence]

def stringlist(train_indices):
    lst1 = []
    lst2 = []
    temp = []
    for i in train_indices:
        for j in i:
            temp = str(j)
            lst1.append(temp)
        lst2.append(lst1)
        lst1 = []
    return lst2

## Modify the language with the language you want to pretrain
lang = Lang("language")
lang.build_vocab(token_train, voc_size=25000)
indice_train = lang.one_hot_trans(token_train)
indice_dev = lang.one_hot_trans(token_dev)

indice_train_str = stringlist(indice_train)
model = gensim.models.Word2Vec(indice_train_str, min_count=0, size = 256, sg=1)
model.save(upload + 'model')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

## Actual order may vary due to different languages
## Need to double check
emb_vec = np.zeros((25004,256))
emb_vec[1,] = model.wv.vectors[1,]
emb_vec[4,] = model.wv.vectors[0,]
emb_vec[5:,] = model.wv.vectors[2:,]
pretrained_weights = emb_vec
pkl.dump(pretrained_weights, open( upload + "pretrained_weights.p", "wb" ))

