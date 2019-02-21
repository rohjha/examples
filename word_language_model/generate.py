###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--dict_data', type=str, default='./data/wikitext-2',
                    help='location of the corpus to get data from')
parser.add_argument('--data', type=str, default='./data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=12,
                    help='sequence length')
parser.add_argument('--num_out', type=int, default=15,
                    help='number of alternatives to show')       
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.dict_data, args.data)
print("yo")

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
print("yo")

# Assumes "<blank>" is one of the words if we need a blank
def tokenize_str(str):
    words = str.split()
    if len(words) >= args.bptt:
        words = words[:args.bptt]
    
    ids = [corpus.dictionary.get_id(word) for word in words]
    return ids

input_original = tokenize_str("He <blank> the room suddenly and went outside to the park <eos>")
input = []
for token in input_original:
    input.append([token])
input = torch.LongTensor(input).to(device)

with torch.no_grad():  # no tracking history
    output, hidden = model(input, hidden)
    word_weights = output.squeeze().exp().to(device)
    
    top_ids = torch.topk(word_weights, args.num_out)[1]
    import pdb; pdb.set_trace()
    for id in top_ids:
        print(corpus.dictionary.idx2word[id])
