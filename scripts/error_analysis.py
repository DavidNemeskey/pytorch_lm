#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads a model and computes the PPL for each individual word, as well as the top
N candidates, the rank of the real word, etc.
"""

import argparse

import torch

from pytorch_lm.bptt import FixNumSteps
from pytorch_lm.data import Corpus, LMData
from pytorch_lm.loss import SequenceLoss


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Loads a model and computes the PPL for each individual '
                    'word, as well as the top N candidates, the rank of the '
                    'real word, etc.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--file', '-f', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='which file to load. Default: test.')
    parser.add_argument('--dont-shuffle', dest='shuffle', action='store_false',
                        help='do not shuffle the sentences in the training set.'
                             'Note that most papers (starting with Zaremba '
                             'et at. (2014)) published results for an '
                             'unshuffled PTB.')
    parser.add_argument('--model', '-m', type=str, default='LSTM',
                        help='the model to load.')
    parser.add_argument('--batch', '-b', type=int, dest='batch_size', default=20,
                        help='the batch size. Default is 20.')
    parser.add_argument('--steps', '-s', type=int, dest='num_steps', default=20,
                        help='the number of timesteps. Default is 20.')
    parser.add_argument('--cuda', '-c', action='store_true', help='use CUDA')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    return parser.parse_args()


def evaluate(model, corpus, data_source, criterion, batch_size, num_steps=1):
    assert num_steps == 1, 'num_steps greater than 1 are not supported'
    # Turn on evaluation mode which disables dropout.
    model.eval()
    steps = FixNumSteps(num_steps)
    data_len = data_source.data.size(1)
    hidden = model.init_hidden(batch_size)
    # for i in range(0, data_len - 1, num_steps):
    for data, targets in data_source.get_batches(steps, evaluation=True):
        print('Data', data.size(), data)
        print('Targets', targets.size(), targets)
        output, hidden = model(data, hidden)
        print('Output', output.size())
        losses = criterion(output, targets).data
        print('Losses', losses.size())
        hidden = repackage_hidden(hidden)
        break


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]


def main():
    args = parse_arguments()

    if torch.cuda.is_available() and not args.cuda:
            logger.warning('You have a CUDA device, so you should probably '
                           'run with --cuda')

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data, args.shuffle)
    vocab_size = len(corpus.dictionary)

    ###############################################################################
    # Load the model, etc.
    ###############################################################################

    # Load the best saved model.
    with open(args.model, 'rb') as f:
        model = torch.load(f)
    criterion = SequenceLoss(reduce_across_batch=None,
                             reduce_across_timesteps=None)
    data = LMData(getattr(corpus, args.file), args.batch_size, args.cuda)
    evaluate(model, corpus, data, criterion, args.batch_size, args.num_steps)


if __name__ == '__main__':
    main()
