""" This file defines modules for looking up embeddings given word ids. """

from os import path
import pickle

import numpy as np
import torch
from torch import nn

# import allennlp.modules.elmo as allen_elmo

# These are the different embedding sizes. Feel free to experiment
# with different sizes for random.
sizes = {"elmo": 1024, "glove": 200, "none": 200}
sizes["both"] = sizes["elmo"] + sizes["glove"]


# TODO: Get Elmo to work
class Elmo(nn.Module):
    """ Finish implementing __init__, forward, and _get_charids for Elmo embeddings.
        Take a look at the Allen AI documentation on using Elmo:
            https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        In particular, reference the section "Using ELMo as a PyTorch Module".
        In addition, the Elmo model documentation may be useful:
            https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L34
    """

    def __init__(self, idx2word, device=torch.device('cuda')):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = allen_elmo.Elmo(options, weights, 1) # initialise an allen_elmo.Elmo model
        self.idx2word = idx2word # Note: you'll need this mapping for _get_charids

        self.embed_size = sizes["elmo"]
        self._dev = device

    def forward(self, batch):
        char_ids = self._get_charids(batch)
        # get elmo embeddings given char_ids:
        return self.elmo(char_ids)['elmo_representations'][0]

    def _get_charids(self, batch):
        """ Given a batch of sentences, return a torch tensor of character ids.
                :param batch: List of sentences - each sentence is a list of int ids
            Return:
                torch tensor on self._dev
        """
        # 1. Map each sentence in batch to a list of string tokens (hint: use idx2word)
        # 2. Use allen_elmo.batch_to_ids to convert sentences to character ids.
        batch_in_words = [[self.idx2word[id] for id in sentence] for sentence in batch]
        return allen_elmo.batch_to_ids(batch_in_words).to(self._dev)

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

class ElmoGlove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
        """ construct Elmo and Glove lookup instances """
        super(ElmoGlove, self).__init__()

        self.elmo = Elmo(idx2word, device)
        self.glove = Glove(data_dir, idx2word, device)

        self.embed_size = sizes["both"]
        self._dev = device

    def forward(self, batch):
        """ Concatenate ELMo and GloVe embeddings together """
        elmo_embeddings = self.elmo.forward(batch)
        glove_embeddings = self.glove.forward(batch)

        embeddings = []
        for elmo_embedding, glove_embedding in zip(elmo_embeddings, glove_embeddings):
            embeddings.append(torch.cat((elmo_embedding, glove_embedding), 1))
        
        return torch.stack(embeddings)


