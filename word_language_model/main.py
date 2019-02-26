# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import random
import embeddings
import pickle
import numpy as np

from torch.utils.data import DataLoader

from tqdm import tqdm

import model
import corpus

# TODO: Add dropout back in

parser = argparse.ArgumentParser(description='Fill-in-the-blank model')
parser.add_argument('--data', type=str, default='/data/course/cs2952d/rjha/data/rsa',
                    help='location of the corpora and embeddings')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers in the encoder')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--embed', type=str, default='None',
                    help="pretrained embeddings to use (None, Glove)")
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

###############################################################################
# Load data
###############################################################################

modes = ['train', 'test', "valid"]

# Note: each of these are dicts that map mode -> object, depending on if we're using the training or dev data.
datasets = {mode: corpus.Corpus(args.data, mode) for mode in modes}
data_sizes = {mode: len(datasets[mode]) for mode in modes} # useful for averaging loss per batch

print("Creating dataloaders")
dataloaders = {mode: DataLoader(datasets[mode], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True) for mode in modes}

###############################################################################
# Build the model
###############################################################################

idx2word = np.load(os.path.join(args.data, "idx2word.npy")).item()
word2idx = np.load(os.path.join(args.data, "word2idx.npy")).item()

# TODO: This is a dumb decision that might make some difference: representing blank with underscore
blank_idx = word2idx['__']

if args.embed.lower() == "glove":
    lookup = embeddings.Glove(args.data, idx2word, device=device)
    emsize = embeddings.sizes["glove"]
else:
    lookup = None
    emsize = embeddings.sizes["none"]

ntokens = len(idx2word)
model = model.RNNModel(args.model, ntokens, emsize, args.nhid, args.nlayers, args.batch_size, args.tied, lookup).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def swap_blanks(batch, seq_lens):
    targets = np.empty(len(batch))
    batch = batch.clone()
    for i in range(len(batch)):
        if (seq_lens[i] == 0):
            continue
        blank_index = random.randint(0, seq_lens[i] - 1)
        targets[i] = batch[i][blank_index]
        batch[i][blank_index] = blank_idx

    return batch, targets

def evaluate(mode):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(idx2word)
    with torch.no_grad():
        for (vectorized_seq, seq_len) in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, args.epochs)):
            data, targets = swap_blanks(vectorized_seq, seq_len)
            data = data.transpose_(0, 1)

            output = model(data, seq_len)

            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, torch.from_numpy(targets).long().cuda()).item()
    return total_loss / (data_sizes[mode] - 1)


def train(epoch):
    # Turn on training mode which enables dropout.
    mode = "train"

    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(idx2word)
    batch = 0

    for (vectorized_seq, seq_len) in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, args.epochs)):
        data, targets = swap_blanks(vectorized_seq, seq_len)
        data = data.transpose_(0, 1)

        model.zero_grad()
        output = model(data, seq_len)

        loss = criterion(output.view(-1, ntokens), torch.from_numpy(targets).long().cuda())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        batch += 1

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, data_sizes[mode] // max_seq_length,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate("valid")
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate("test")
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
