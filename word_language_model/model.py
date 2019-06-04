import torch
import torch.nn as nn
import embeddings
from torch.autograd import Variable
import pdb
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNEncoder(nn.Module):
    def __init__(self, ninp, nhid, nlayers, bsz):
        super(RNNEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers)
        self.nlayers = nlayers
        self.nhid = nhid

    def forward(self, emb, seq_len, hidden):
        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        _, hidden = self.rnn(packed_input, hidden)
        
        _, unperm_idx = perm_idx.sort(0)
        return hidden[0][1][unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class AttentionEncoder(nn.Module):
    def __init__(self, ninp, nhid, nlayers, max_seq_length):
        super(AttentionEncoder, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers)
        self.attn = torch.nn.Linear(self.nhid, max_seq_length)

    def forward(self, emb, seq_len, hidden):
        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        output, hidden = self.rnn(packed_input, hidden)
        output, lengths = pad_packed_sequence(output)
        hidden = hidden[0][1]

        attn_weights = F.softmax(self.attn(hidden), dim=1)
        max_length = torch.max(lengths).item()
        attn = torch.bmm(attn_weights[:, :max_length].unsqueeze(0).transpose(0, 1), output.transpose(0, 1))

        attn = F.relu(attn)
        _, unperm_idx = perm_idx.sort(0)
        return attn[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class AttentionEncoderV2(nn.Module):
    def __init__(self, ninp, nhid, nlayers, max_seq_length):
        super(AttentionEncoderV2, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers)
        self.attn = torch.nn.Linear(self.nhid, self.nhid)

    def forward(self, emb, seq_len, hidden):
        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(output)
        hidden = hidden[0][1]

        pre_attn_weights = F.softmax(self.attn(hidden), dim=1)
        attn_weights = torch.bmm(pre_attn_weights.unsqueeze(1), output.transpose(0, 1).transpose(1, 2))
        attn = torch.bmm(attn_weights, output.transpose(0, 1)).squeeze(1)

        attn = F.relu(attn)
        _, unperm_idx = perm_idx.sort(0)
        return attn[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class Decoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoded):
        return self.decoder(encoded)
  
class SimpleContainer(nn.Module):
    def __init__(self, embed, encoder, decoder):
        super(SimpleContainer, self).__init__()
        self.embed = embed
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input, seq_len, hidden):
        emb = (self.embed(input).detach())
        encoded = self.encoder(emb, seq_len, hidden)
        return self.decoder(encoded)

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)
