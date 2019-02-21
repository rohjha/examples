import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_id(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, dict_path, path):
        self.dictionary = Dictionary()
        self.fill_dict(os.path.join(dict_path, 'train.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def fill_dict(self, dict_path):
        assert os.path.exists(dict_path)
        # Add words to the dictionary
        with open(dict_path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)

        self.dictionary.add_word('<blank>')
        print(len(self.dictionary))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.get_id(word)
                    token += 1
        print("yo")

        return ids
