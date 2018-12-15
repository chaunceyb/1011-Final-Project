import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, args, cons, input_size, hidden_size, n_layers=1, dropout=0.2):
        super(EncoderRNN, self).__init__()

        global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
        device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons

        self.input_size = input_size
        self.hidden_size = hidden_size // 2
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = args.rnn

        # Train Epochs
        self.epochs = 0

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, self.hidden_size, n_layers,
                              dropout=self.dropout, bidirectional=True, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, self.hidden_size, n_layers,
                              dropout=self.dropout, bidirectional=True, batch_first=True)

    def initHidden(self, batch_size, batch_seq):
        if self.rnn_type == 'gru':
            return torch.zeros(2 * self.n_layers, batch_size, self.hidden_size).to(device)
        elif self.rnn_type == 'lstm':
            return (torch.zeros(2 * self.n_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(2 * self.n_layers, batch_size, self.hidden_size).to(device))

    def forward(self, input_seqs, input_lengths, hidden=None):
        batch_size = input_seqs.size(0)
        batch_seq = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        if hidden == None:
            hidden = self.initHidden(batch_size, batch_seq)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                            batch_first=True)  # unpack (back to padded) # S x B x H
        # IMPORTANT: the pad_packed function also needs batch_first parameters
        if self.rnn_type == 'gru':
            hidden = self.transform(hidden, self.n_layers)

        elif self.rnn_type == 'lstm':
            hidden = [self.transform(h, self.n_layers) for h in hidden]
            # print(type(hidden))

        return outputs, hidden  # HIDDEN: L x B x H

    def transform(self, x, num_layers):
        even = (torch.LongTensor(range(num_layers)) + 1) * 2 - 1
        odd = even - 1
        even.to(device)
        odd.to(device)
        return torch.cat((x[odd], x[even]), dim=2)

