import torch
import torch.nn as nn
import embeddings
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# TODO: Use ELMo

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, bsz, tie_weights=False, embed=None):
        super(RNNModel, self).__init__()
        self.bsz = bsz

        if embed is None:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = embed

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        from_scratch = embed == None
        self.init_weights(from_scratch)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, from_scratch):
        initrange = 0.1
        if (from_scratch):
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, seq_len, hidden):
        emb = (self.encoder(input)).detach()

        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:,perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        _, hidden = self.rnn(packed_input, hidden)
        decoded = self.decoder(hidden[-1][1])

        _, unperm_idx = perm_idx.sort(0)
        return decoded[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)
