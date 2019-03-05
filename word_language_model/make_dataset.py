import argparse
import os
import random

parser = argparse.ArgumentParser(description='Building Wiki dataset')
parser.add_argument('--directory', type=str, default='/data/course/cs2952d/rjha/data/rsa/',
                    help='location of the directory to store the final corpus files')
parser.add_argument('--corpus_location', type=str, default='/data/nlp/atomicwikiedits/en/final-sentences-only.txt',
                    help='location (including filename) of the original corpus')
parser.add_argument('--ratio', type=float, default=0.0006,
                    help='proportion of sentences to put in test and validation')
parser.add_argument('--seed', type=int, default=29,
                    help='random seed to use')
parser.add_argument('--max_seq_length', type=int, default=35,
                    help='if a sentence is longer than this, it will be thrown out')
parser.add_argument('--max_seq_length', type=int, default=5,
                    help='if a sentence is longer than this, it will be thrown out')
args = parser.parse_args()

assert os.path.exists(args.corpus_location)
assert os.path.exists(args.directory)

print("starting")
with open(args.corpus_location, 'r') as orig, \
     open(os.path.join(args.directory, 'train.txt'), 'w+') as train, \
     open(os.path.join(args.directory, 'test.txt'), 'w+') as test, \
     open(os.path.join(args.directory, 'valid.txt'), 'w+') as valid:

    random.seed(args.seed)

    line_number = 0
    thrown_out = 0
    for line in orig:
        line_number += 1
        if line_number % 1000000 == 0:
            print("finished {}m lines".format(line_number / 1000000))

        if '|' in line:
            thrown_out += 1
            continue

        words = line.lower().strip('\n').split(' ')
        orig_len = len(words)
        if orig_len > args.max_seq_length or orig_len < args.min_seq_length:
            thrown_out += 1
            continue

        for i in range(args.max_seq_length - len(words)):
            words.append('<pad>')

        out_line = " ".join(words) + "|" + str(orig_len) + "\n"
        rand = random.random()

        if rand < args.ratio:
            test.write(out_line)
        elif rand < 2 * args.ratio:
            valid.write(out_line)
        else:
            train.write(out_line)

print("threw out {}/{} lines".format(thrown_out, line_number))
print("done")