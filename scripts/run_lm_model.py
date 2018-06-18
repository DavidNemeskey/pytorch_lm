#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic language model training script."""

import argparse
from functools import partial
import json
import math
import random
import time

import torch
from torch.autograd import Variable

from pytorch_lm.bptt import create_num_steps
from pytorch_lm.utils.config import create_object, create_function
from pytorch_lm.utils.config import get_config_file
from pytorch_lm.data import Corpus, LMData
from pytorch_lm.loss import SequenceLoss
from pytorch_lm.lr_schedule import lr_step_at_epoch_start, ConstantLR
from pytorch_lm.utils.lang import getall
from pytorch_lm.utils.logging import setup_stream_logger


logger = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generic language model training script. It can train a '
                    'selection of language models, starting with (the correct '
                    'reimplementation of) Zaremba et al. (2014).\n\n'
                    'Based on the PyTorch Wikitext-2 RNN/LSTM Language Model.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--dont-shuffle', dest='shuffle', action='store_false',
                        help='do not shuffle the sentences in the training set.'
                             'Note that most papers (starting with Zaremba '
                             'et at. (2014)) published results for an '
                             'unshuffled PTB.')
    parser.add_argument('--model', '-m', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--early-stop', '-e', type=int, default=0,
                        help='early stopping: stop if the validation loss does '
                             'not improve for this many epochs.')
    parser.add_argument('--seed', '-s', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', '-c', action='store_true', help='use CUDA')
    parser.add_argument('--config-file', '-C', required=True,
                        help='the configuration file.')
    parser.add_argument('--save', '-S', type=str,
                        help='the name of the file to save the model to (if any)')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    parser.add_argument('--log-interval', '-I', type=int, default=200, metavar='N',
                        help='report interval')
    return parser.parse_args()


def read_config(config_file, vocab_size):
    """
    Reads the configuration file, and creates the model, optimizer and
    learning rate schedule objects used by the training process.
    """
    with open(get_config_file(config_file)) as inf:
        config = json.load(inf)
    train = config.pop('train', {})
    valid = config.pop('valid', {})
    test = config.pop('test', {})

    # Copy all keys from the main dictionary to the sub-dictionaries, but do
    # not overwrite keys already there
    for k, v in config.items():
        for d in [train, valid, test]:
            if k not in d:
                d[k] = v

    # Now for the model & stuff (train only)
    try:
        train['model'] = create_object(train['model'],
                                       base_module='pytorch_lm.model',
                                       args=[vocab_size])
    except KeyError:
        raise KeyError('Missing configuration key: "model".')
    try:
        train['optimizer'] = create_object(train['optimizer'],
                                           base_module='torch.optim',
                                           args=[train['model'].parameters()])
    except KeyError:
        raise KeyError('Missing configuration key: "optimizer".')
    try:
        train['initializer'] = create_function(train['initializer'],
                                               base_module='torch.nn.init')
    except KeyError:
        raise KeyError('Missing configuration key: "initializer".')
    if 'bias_initializer' in train:
        train['bias_initializer'] = create_function(train['bias_initializer'],
                                                    base_module='torch.nn.init')
    if 'lr_scheduler' in train:
        train['lr_scheduler'] = create_object(train['lr_scheduler'],
                                              base_module='pytorch_lm.lr_schedule',
                                              args=[train['optimizer']])
    else:
        train['lr_scheduler'] = ConstantLR(train['optimizer'])

    full_config = {'train': train, 'valid': valid, 'test': test}

    # Initialization of stuff in train, valid and test
    for sub_config in full_config.values():
        if 'num_steps' in sub_config:
            sub_config['num_steps'] = create_num_steps(sub_config['num_steps'])

    return full_config


def train(model, corpus, config, train_data, criterion, epoch, log_interval):
    optimizer, batch_size, num_steps, grad_clip = getall(
        config, ['optimizer', 'batch_size', 'num_steps', 'grad_clip'])
    # Turn on training mode which enables dropout.
    model.train()

    # lr = lr_scheduler.get_lr()[0]
    lr = optimizer.param_groups[0]['lr']
    total_loss = 0
    start_time = time.time()
    data_len = train_data.data.size(1)
    hidden = model.init_hidden(batch_size)

    for batch, (data, targets, lr_ratio) in enumerate(train_data.get_batches(num_steps)):
        seq_len = targets.size(1)
        # def to_str(f):
        #     return corpus.dictionary.idx2word[f]

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        # TODO encapsulate this somewhere
        # For random BPTT length
        optimizer.param_groups[0]['lr'] = lr * lr_ratio

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output, targets) + model.loss_regularizer()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # for name, p in model.named_parameters():
        #     print('GRAD', name, p.grad.data)

        optimizer.step()
        # for name, p in model.named_parameters():
        #     p.data.add_(-1 * lr, p.grad.data)

        # For random BPTT length
        optimizer.param_groups[0]['lr'] = lr

        total_loss += loss.data / seq_len

        if batch % log_interval == 0 and batch > 0:
            # cur_loss = total_loss[0] / log_interval
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                        'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, data_len // num_steps.len, lr,
                            elapsed * 1000 / log_interval, cur_loss,
                            math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model, corpus, data_source, criterion, batch_size, num_steps):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    data_len = data_source.data.size(1)
    hidden = model.init_hidden(batch_size)
    # for i in range(0, data_len - 1, num_steps):
    for data, targets in data_source.get_batches(num_steps, evaluation=True):
        output, hidden = model(data, hidden)
        cost = criterion(output, targets).data
        total_loss += cost
        hidden = repackage_hidden(hidden)
    # return total_loss[0] / data_len
    return total_loss.item() / data_len


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]

def initialize_model(model, initializer, bias_initializer=None):
    """
    Recursively initializes all parameters of the model. It accepts two
    initializer functions: one for (main) weights and one for biases. The
    latter defaults to constant zero.
    """
    if not bias_initializer:
        bias_initializer = partial(torch.nn.init.constant_, val=0)
    for name, p in model.named_parameters():
        if name.lower().endswith('bias'):
            bias_initializer(p.data)
        else:
            initializer(p.data)


def main():
    args = parse_arguments()

    global logger
    logger = setup_stream_logger(args.log_level)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning('You have a CUDA device, so you should probably '
                           'run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data, args.shuffle)
    vocab_size = len(corpus.dictionary)

    ###############################################################################
    # Build the model, etc.
    ###############################################################################

    config = read_config(args.config_file, vocab_size)
    traind, validd, testd = getall(config, ['train', 'valid', 'test'])
    train_data = LMData(corpus.train, traind['batch_size'], args.cuda)
    valid_data = LMData(corpus.valid, validd['batch_size'], args.cuda)
    test_data = LMData(corpus.test, testd['batch_size'], args.cuda)

    model, optimizer, initializer, bias_initializer, lr_scheduler = getall(
        traind, ['model', 'optimizer', 'initializer',
                 'bias_initializer', 'lr_scheduler'])

    logger.info('Model object created: {}'.format(model))

    initialize_model(model, initializer)

    # model.double()
    if args.cuda:
        model.cuda()

    ###############################################################################
    # Training code
    ###############################################################################

    # criterion = nn.CrossEntropyLoss()
    criterion = SequenceLoss(reduce_across_batch='mean',
                             reduce_across_timesteps='sum')

    # Loop over epochs.
    best_val_loss = None
    not_improved_for = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, traind['num_epochs'] + 1):
            epoch_start_time = time.time()
            if lr_step_at_epoch_start(lr_scheduler):
                lr_scheduler.step()
            train(model, corpus, traind, train_data, criterion,
                  epoch, args.log_interval)
            val_loss = evaluate(model, corpus, valid_data,
                                criterion, validd['batch_size'],
                                validd['num_steps'])
            if not lr_step_at_epoch_start(lr_scheduler):
                lr_scheduler.step(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | '
                        'valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                            epoch, (time.time() - epoch_start_time),
                            val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
                not_improved_for = 0
            else:
                not_improved_for += 1

            if args.early_stop > 0 and not_improved_for == args.early_stop:
                logger.info('Validation performance has not improved for '
                            '{} epochs; stopping early'.format(not_improved_for))
                break
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, corpus, test_data,
                         criterion, testd['batch_size'], testd['num_steps'])
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('-' * 89)


if __name__ == '__main__':
    main()
