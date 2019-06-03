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
    
class RNNContainer(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, bsz, embed):
        super(RNNContainer, self).__init__()
        self.embed = embed
        self.encoder = RNNEncoder(ninp, nhid, nlayers, bsz)
        self.decoder = Decoder(nhid, ntoken)

    def forward(self, input, seq_len, hidden):
        emb = (self.embed(input).detach())
        encoded = self.encoder(emb)
        return self.decoder(encoded)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    # NOTE: Need to comment out dropout when testing older models
    def __init__(self, ntoken, ninp, nhid, nlayers, bsz, embed, dropout=0):
        super(RNNModel, self).__init__()
        self.bsz = bsz
        self.encoder = embed

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers)

        self.decoder = nn.Linear(nhid, ntoken)
        self.drop = nn.Dropout(dropout)
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

        output, hidden = self.rnn(packed_input, hidden)
        decoded = self.decoder(hidden[0][1])

        _, unperm_idx = perm_idx.sort(0)
        return decoded[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class AttentionModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder that has attention and another recurrent model."""

    # TODO: Add dropout
    def __init__(self, ntoken, ninp, nhid, nlayers, bsz, embed, max_seq_length, emb_size, dropout=0):
        super(AttentionModel, self).__init__()
        self.bsz = bsz
        self.max_seq_length = max_seq_length
        self.emb_size = emb_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken

        self.embed = embed
        self.encoder = torch.nn.LSTM(ninp, nhid, nlayers)

        self.attn = torch.nn.Linear(self.nhid, self.max_seq_length)
        self.out = torch.nn.Linear(self.nhid, self.ntoken)

    def forward(self, input, seq_len, hidden):
        emb = (self.embed(input).detach())

        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        output, hidden = self.encoder(packed_input, hidden)
        output, lengths = pad_packed_sequence(output)
        hidden = hidden[0][1]

        attn_weights = F.softmax(self.attn(hidden), dim=1)
        max_length = torch.max(lengths).item()
        attn = torch.bmm(attn_weights[:, :max_length].unsqueeze(0).transpose(0, 1), output.transpose(0, 1))

        attn = F.relu(attn)

        decoded = self.out(attn)
        _, unperm_idx = perm_idx.sort(0)
        return decoded[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class AttentionModelV2(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder that has attention and another recurrent model."""

    # TODO: Add dropout
    def __init__(self, ntoken, ninp, nhid, nlayers, bsz, embed, max_seq_length, emb_size, dropout=0):
        super(AttentionModelV2, self).__init__()
        self.bsz = bsz
        self.max_seq_length = max_seq_length
        self.emb_size = emb_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken

        self.embed = embed
        self.encoder = torch.nn.LSTM(ninp, nhid, nlayers)

        self.attn = torch.nn.Linear(self.nhid, self.nhid)
        self.out = torch.nn.Linear(self.nhid, self.ntoken)

    def forward(self, input, seq_len, hidden):
        emb = (self.embed(input).detach())

        seq_len, perm_idx = seq_len.sort(0, descending=True)
        emb = emb[:, perm_idx]

        packed_input = pack_padded_sequence(emb, seq_len.int().cpu().numpy())

        output, hidden = self.encoder(packed_input, hidden)
        output, _ = pad_packed_sequence(output)
        hidden = hidden[0][1]

        pre_attn_weights = F.softmax(self.attn(hidden), dim=1)
        attn_weights = torch.bmm(pre_attn_weights.unsqueeze(1), output.transpose(0, 1).transpose(1, 2))
        attn = torch.bmm(attn_weights, output.transpose(0, 1)).squeeze(1)

        attn = F.relu(attn)

        decoded = self.out(attn)
        _, unperm_idx = perm_idx.sort(0)
        return decoded[unperm_idx]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

class BERTModel(nn.Module):
    """Simple model with BERT, followed by a feed-forward layer."""

    def __init__(self, ntoken, nhid, emsize, bsz, embed):
        super(BERTModel, self).__init__()
        self.bsz = bsz
        self.encoder = embed # this will be BERT

        self.first_linear = nn.Linear(emsize, nhid)
        self.second_linear = nn.Linear(nhid, ntoken)
        self.decoder = nn.Sequential(self.first_linear, nn.ReLU(), self.second_linear)
        self.emsize = emsize

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.first_linear.bias.data.zero_()
        self.first_linear.weight.data.uniform_(-initrange, initrange)

        self.second_linear.bias.data.zero_()
        self.second_linear.weight.data.uniform_(-initrange, initrange)

    # Sadly need to switch between commented and uncommented when going from training to testing
    def forward(self, input, blank_indices):
        bsz = 1 # self.bsz
 
        emb = (self.encoder(input).detach())
        blank_embeds = torch.gather(
            emb[blank_indices], 
            1, 
            torch.arange(0, bsz).cuda().view(-1, 1).unsqueeze(2).repeat(1, 1, self.emsize))
        blank_embeds = blank_embeds.view(bsz, self.emsize).cuda()
        return self.decoder(blank_embeds)