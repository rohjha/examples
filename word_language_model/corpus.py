""" This file defines a pytorch Dataset class for loading Wikipedia data, and applying
    our specific preprocessing (mapping word tokens to integer ids)
"""

from os import path
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class Corpus(Dataset):
    max_seq_length = 35

    """
    This is the Dataset class for this net. When initializing the dataset we
    specify mode "train", "dev", or "test".
    """
    def __init__(self, data_dir, mode):
        print("Loading dataset for {}".format(mode))

        print("Loading word2idx into Numpy")
        self.word2idx = np.load(path.join(data_dir, "word2idx.npy")).item()
        self.unk_idx = self.word2idx['<unk>']

        print("Loaded {} words".format(len(self.word2idx)))

        with open(path.join(data_dir, mode+".txt")) as f:
            line_count = 0
            for line in f:
                line_count += 1
            
            self.seqs = np.empty((line_count, self.max_seq_length))
            self.lens = np.empty(line_count)

        with open(path.join(data_dir, mode+".txt")) as f:
            i = 0
            for line in f:
                line_split = line.split('|')
                words = line_split[0].split(' ')
                self.seqs[i] = np.asarray([self.get_id(word) for word in words])
                self.lens[i] = int(line_split[1]) 
                i += 1

                if i % 500000 == 0:
                    print("finished {}m lines".format(i / 1000000))
            
            print("Done with {}".format(mode))

        self.seqs = self.seqs.astype(int)


    def get_id(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.unk_idx


    def __len__(self):
        """ Returns the length of the data (number of sentences) """
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        """ Given an index, returns the vectorized sentence and the original length
            of that sentence
        """
        return (self.seqs[idx], self.lens[idx])
