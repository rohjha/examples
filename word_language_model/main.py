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

from torch.utils.data import DataLoader

from tqdm import tqdm

import corpus
import model

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
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
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
train_index = 0; test_index = 1; valid_index = 2

# Note: each of these are dicts that map mode -> object, depending on if we're using the training or dev data.
datasets = {mode: corpus.Corpus(args.data, mode) for mode in modes}
data_sizes = {mode: len(datasets[mode]) for mode in modes} # useful for averaging loss per batch
dataloaders = {mode: DataLoader(datasets[mode], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True) for mode in modes}

###############################################################################
# Build the model
###############################################################################

with open(os.path.join(args.data, "idx2word.dict"), "rb") as f:
    idx2word = pickle.load(f)

if args.embed.lower() == "glove":
    lookup = embeddings.Glove(args.glove_data, idx2word, device=device)
    emsize = embeddings.sizes["glove"]
else:
    lookup = None
    emsize = embeddings.sizes["none"]

ntokens = len(idx2word)
model = model.RNNModel(args.model, ntokens, emsize, args.nhid, args.nlayers, args.dropout, args.tied, lookup).to(device)
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

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    blank_index = random.randint(0, seq_len-1)
    target = data[blank_index].clone()
    data[blank_index].fill_(len(idx2word) - 1)
    return data, target

def swap_blanks(batch, seq_lens):
    return batch, None

def evaluate(mode_index):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(idx2word)
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for (vectorized_seq, seq_len) in tqdm(dataloaders[mode_index], desc='{}:{}/{}'.format(modes[mode_index], epoch, args.epochs)):
            data, targets = get_batch(vectorized_seq, seq_len)
            output, hidden = model(data, hidden)
            output = output

            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (data_sizes[mode_index] - 1)


def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(idx2word)
    hidden = model.init_hidden(args.batch_size)

    batch = 0
    for (vectorized_seq, seq_len) in tqdm(dataloaders[train_index], desc='{}:{}/{}'.format(modes[train_index], epoch, args.epochs)):
        data, targets = get_batch(vectorized_seq, seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        # import pdb; pdb.set_trace()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
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
                epoch, batch, data_sizes[train_index] // args.bptt,
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
        val_loss = evaluate(valid_index)
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
test_loss = evaluate(test_index)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
