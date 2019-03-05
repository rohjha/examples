###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import numpy as np
import torch
import os

import corpus

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='/data/course/cs2952d/rjha/data/rsa',
                    help='location of the corpora and embeddings')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--num_out', type=int, default=15,
                    help='number of alternatives to show')       
args = parser.parse_args()

# This is a function of the data file. Need to use make_dataset.py to make data files with 
# a different maximal length.
max_seq_length = 35

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

idx2word = np.load(os.path.join(args.data, "idx2word.npy")).item()
word2idx = np.load(os.path.join(args.data, "word2idx.npy")).item()

ntokens = len(idx2word)
unk_idx = word2idx["<unk>"]

# Assumes "__" is one of the words if we need a blank
def tokenize_str(str):
    words = str.split()
    if len(words) >= max_seq_length:
        words = words[:max_seq_length]
    
    ids = [word2idx.get(word, unk_idx) for word in words]
    return ids

input_original = tokenize_str("He likes to __ waffles off his plate .")
input = []
for token in input_original:
    input.append([token])
input = torch.LongTensor(input).to(device)

with torch.no_grad():  # no tracking history
    output = model(input, torch.LongTensor(np.asarray([len(input)])).to(device), model.init_hidden(1))
    word_weights = output.squeeze().to(device)
    
    res = torch.topk(word_weights, args.num_out)
    top_ids = res[1]
    top_props = res[0]

    for i in range(args.num_out):
        print("%s, %s" % (idx2word[top_ids.data[i].item()], top_props[i]))
