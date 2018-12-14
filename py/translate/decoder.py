import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def  __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        if self.method == 'class':
            self.lin1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.lin2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.lin3 = nn.Linear(self.hidden_size, 1)

    def forward(self, rnn_output, encoder_outputs, x_mask):

        attn_energies = self.score(rnn_output, encoder_outputs)  # -> B x tgt_len x src_len

        attn_energies.masked_fill_(1 - x_mask.unsqueeze(1), -float('inf'))

        return F.softmax(attn_energies, dim=2)

    def score(self, rnn_output, encoder_output):

        # TODO rewrite class to fit tgt_len
        if self.method == 'class':
            raise NotImplementedError
            assert rnn_output.shape[0] == encoder_output.shape[0]
            rnn_output = self.lin1(rnn_output)
            encoder_output = self.lin2(encoder_output)
            energy = torch.tanh(rnn_output.add(encoder_output))  # Hi PyTorch, please broadcast hidden to B x S x 2N
            energy = self.lin3(energy)
            return energy  # B x S x 1

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            W = self.attn(encoder_output)  # B x src_len x H
            score = torch.bmm(rnn_output, W.transpose(1, 2))
            return score

class AttnDecoderRNN(nn.Module):
    def __init__(self, args, cons, attn_model, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(AttnDecoderRNN, self).__init__()
        global device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
        device, BATCH_SIZE, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = cons

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = args.rnn

        # Train Epochs
        self.epochs = 0

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.embedding_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        # TODO Middle size
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, encoder_hidden, x_mask):
        #             B x tgt_len  L x B x H   B x src_len x H  L x B x H
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.rnn(embedded, last_hidden)
        # B x tgt_len x H, L x B x H
        if self.attn_model != 'none':
            # Attention
            attn_weights = self.attn(rnn_output, encoder_outputs, x_mask)  # B x tgt_len x src_len
            # B x tgt_len x H   B x src_len x H
            context = attn_weights.bmm(encoder_outputs)  # B x tgt_len x H
        else:
            encoder_hidden = (encoder_hidden[self.n_layers - 1]).unsqueeze(1).repeat(1, input_seq.size(1), 1)# should be top layer, right?
            context = encoder_hidden
        concat_input = torch.cat((rnn_output, context), 2)  # B x tgt_len x 2H
        # FC 1
        concat_output = torch.tanh(self.fc1(concat_input))
        concat_output = self.dropout(concat_output)
        # FC 2
        output = self.fc2(concat_output)  # B x tgt_len x output_vocab
        output = torch.log_softmax(output, dim=2)

        # Return final output, hidden state, and attention weights (for visualization)
        if self.attn_model != 'none':
            return output, hidden, attn_weights
        else:
            return output, hidden, 0
