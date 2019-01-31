#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads a model and computes the PPL for each individual word, as well as the top
N candidates, the rank of the real word, etc.
"""

import argparse
from math import exp

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
    context = [[] for _ in range(batch_size)]
    # for i in range(0, data_len - 1, num_steps):
    total_loss = 0
    for data, targets in data_source.get_batches(steps, evaluation=True):
        output, hidden = model(data, hidden)
        # TODO: mondatkezdo
        sorted_logits, most_probable = torch.sort(output, dim=2, descending=True)
        eq_target = (most_probable == targets.unsqueeze(2))
        nnz = eq_target.nonzero()
        indices = targets.index_put((nnz[:, 0], nnz[:, 1]), nnz[:, 2])
        losses = criterion(output, targets).data.view(batch_size, num_steps)
        probabilities = F.softmax(sorted_logits, dim=2)
        coordx, coordy = torch.meshgrid([torch.arange(batch_size), torch.arange(num_steps)])
        target_probs = probabilities[coordx, coordy, indices]
        predicted_probs = probabilities[:, :, 0]
        entropy = (-probabilities * F.log_softmax(sorted_logits, dim=2)).sum(2)
        total_loss += torch.sum(losses).item()
        hidden = repackage_hidden(hidden)
        print('target_word', 'context', 'loss', 'perplexity', 'entropy',
              'target_index', 'target_p', 'predicted_p', 'most_probable',
              sep='\t')
        for i in range(data.size(0)):
            context[i] = (context[i] + [corpus.dictionary.idx2word[data[i, 0]]])[-10:]
            # word, context, loss, perplexity, entropy of the distribution,
            # index of the target word, probability of target word,
            # probability of predicted word, most probable words
            print(corpus.dictionary.idx2word[targets[i, 0]],
                  ' '.join(context[i]),
                  losses[i, 0].item(),
                  exp(losses[i, 0]),
                  entropy[i, 0].item(),
                  indices[i, 0].item(),
                  target_probs[i, 0].item(),
                  predicted_probs[i, 0].item(),
                  ' '.join(corpus.dictionary.idx2word[w] for w in most_probable[i, 0, :5]),
                  sep='\t')
    return total_loss / batch_size / data_len


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
