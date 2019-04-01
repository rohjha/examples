""" This file defines modules for looking up embeddings given word ids. """

from os import path
import pickle

import numpy as np
import torch
from torch import nn

from flair.embeddings import BertEmbeddings
from flair.data import Sentence

# import allennlp.modules.elmo as allen_elmo

# These are the different embedding sizes. Feel free to experiment
# with different sizes for random.
# TODO: add BERT's size
sizes = {"glove": 200, "bert": 768}

class Bert(nn.Module):
    def __init__(self, idx2word, device=torch.device('cpu')):
        super(Bert, self).__init__()
        self.idx2word = idx2word
        self.embed_size = sizes["bert"]
        self.bert = BertEmbeddings('bert-base-uncased', '-2')
    
    def forward(self, batch):
        # TODO: fill this in
        batch_as_words = [[self.idx2word[token] for token in l] for l in batch.transpose(0, 1).tolist()]
        batch_as_sentences = [Sentence(' '.join(l)) for l in batch_as_words]
        embeds = self.bert.embed(batch_as_sentences)
        embeds = [[token.embedding for token in sentence] for sentence in embeds]
        return torch.stack([torch.stack(sentence) for sentence in embeds]).transpose(0, 1).cuda()


class Glove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
        """ load pre-trained GloVe embeddings from disk """
        super(Glove, self).__init__()
        # 1. Load glove.6B.200d.npy from inside data_dir into a numpy array
        #    (hint: np.load)
        # 2. Load glove_tok2id.dict from inside data_dir. This is used to map
        #    a word token (as str) to glove vocab id (hint: pickle.load)
        with open(path.join(data_dir, "glove.6B.200d.npy"), "rb") as f:
            embeddings_np = np.load(f)

        with open(path.join(data_dir, "glove_tok2id.dict"), "rb") as f:
            self.glove_vocab = pickle.load(f)

        self.idx2word = idx2word
        self.embed_size = sizes["glove"]

        # This is hacky but probably okay
        embeddings_np[self.glove_vocab['__']] = np.zeros(200)

        # 3. Create a torch tensor of the glove vectors and construct a
        #    a nn.Embedding out of it (hint: see how RandEmbed does it)
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embeddings_np)) # nn.Embedding layer
        for param in self.embeddings.parameters():
            param.requires_grad = False

        self._dev = device

    def _lookup_glove(self, word_id):
        # given a word_id, convert to string and get glove id from the string:
        # unk if necessary.
        return self.glove_vocab.get(self.idx2word[word_id].lower(), self.glove_vocab["unk"])

    def _get_gloveids(self, batch):
        # import pdb; pdb.set_trace()
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.apply_
        return batch.to(torch.device('cpu')).apply_(self._lookup_glove).to(self._dev)

    def forward(self, batch):
        return self.embeddings(self._get_gloveids(batch))


