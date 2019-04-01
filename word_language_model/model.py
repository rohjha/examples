import torch
import torch.nn as nn
import embeddings
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This is the old joint model
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, bsz, embed, bidir=False):
        super(RNNModel, self).__init__()
        self.bsz = bsz
        self.encoder = embed

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers, bidirectional=bidir)

        if bidir:
            self.decoder = nn.Linear(2 * nhid, ntoken)
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        self.bidir = bidir
        #self.drop = nn.Dropout(dropout)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, seq_len, hidden):
        emb = (self.encoder(input).detach())

        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        _, hidden = self.rnn(packed_input, hidden)

        if self.bidir:
            # NOTE: Unfortunately need to change 'self.bsz' to one when testing bidirectional models
            hidden = hidden[0].view(self.nlayers, 2, self.bsz, self.nhid)
            hidden_forward = hidden[1][0]
            hidden_backward = hidden[1][1]
            decoded = self.decoder(torch.cat((hidden_forward, hidden_backward), 1))
        else:
            decoded = self.decoder(hidden[0][1])

        _, unperm_idx = perm_idx.sort(0)
        return decoded[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.bidir:
            return (weight.new_zeros(self.nlayers * 2, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers * 2, batch_size, self.nhid))
        else:
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))