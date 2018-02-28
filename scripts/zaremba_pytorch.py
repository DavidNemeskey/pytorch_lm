#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implements the small model from Zaremba (2014).
"""

import argparse
import json
import math
import time

import torch
from torch.autograd import Variable

from pytorch_lm.data import Corpus, batchify, get_batch
from pytorch_lm.loss import SequenceLoss
from pytorch_lm.lr_schedule import ConstantLR
from pytorch_lm.utils.config import get_config_file, create_object
from pytorch_lm.utils.logging import setup_stream_logger


logger = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Modification of the PyTorch Wikitext-2 RNN/LSTM Language '
                    'Model, so that it actually does what Zaremba (2014) '
                    'described.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--model', '-m', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--seed', '-s', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', '-c', action='store_true', help='use CUDA')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    parser.add_argument('--log-interval', '-I', type=int, default=200, metavar='N',
                        help='report interval')
    return parser.parse_args()


def train(model, corpus, train_data, criterion, optimizer, lr_schedule,
          epoch, batch_size, num_steps, log_interval):
    # Turn on training mode which enables dropout.
    model.train()
    lr_schedule.step()
    lr = lr_schedule.get_lr()[0]
    total_loss = 0
    start_time = time.time()
    data_len = train_data.size(1)
    hidden = model.init_hidden(batch_size)

    for batch, i in enumerate(range(0, data_len - 1, num_steps)):
        data, targets = get_batch(train_data, i, num_steps)

        def to_str(f):
            return corpus.dictionary.idx2word[f]

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)

        optimizer.step()
        # for name, p in model.named_parameters():
        #     p.data.add_(-1 * lr, p.grad.data)

        total_loss += loss.data / num_steps

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                        'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, data_len // num_steps, lr,
                            elapsed * 1000 / log_interval, cur_loss,
                            math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model, corpus, data_source, criterion, batch_size, num_steps):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    data_len = data_source.size(1)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_len - 1, num_steps):
        data, targets = get_batch(data_source, i, num_steps, evaluation=True)
        output, hidden = model(data, hidden)
        cost = criterion(output, targets).data
        total_loss += cost
        hidden = repackage_hidden(hidden)
    return total_loss[0] / data_len


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]


def read_config(config_file, vocab_size):
    """
    Reads the configuration file, and creates the model, optimizer and
    learning rate schedule objects used by the training process.
    """
    with open(get_config_file(config_file)) as inf:
        config = json.load(inf)
    model = create_object(config['model'], base_module='pytorch_lm.model',
                          args=[vocab_size])
    optimizer = create_object(config['optimizer'], base_module='torch.optim')
    if 'lr_scheduler' in config:
        lr_scheduler = create_object(config['lr_scheduler'],
                                     base_module='pytorch_lm.lr_schedule',
                                     args=[optimizer])
    else:
        lr_scheduler = ConstantLR(optimizer)
    return (model, optimizer, lr_scheduler, config['num_epochs'],
            config['batch_size'], config['num_steps'])


def main():
    args = parse_arguments()

    global logger
    logger = setup_stream_logger(args.log_level)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning('You have a CUDA device, so you should probably '
                           'run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data)
    vocab_size = len(corpus.dictionary)

    ###############################################################################
    # Build the model, etc.
    ###############################################################################

    (model, optimizer, lr_schedule,
     num_epochs, batch_size, num_steps) = read_config(args.config_file, vocab_size)
    train_batch_size = eval_batch_size = batch_size
    train_data = batchify(corpus.train, train_batch_size, args.cuda)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)

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
    # best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            train(model, corpus, train_data, criterion, optimizer, lr_schedule,
                  epoch, train_batch_size, num_steps, args.log_interval)
            val_loss = evaluate(model, corpus, val_data,
                                criterion, eval_batch_size, num_steps)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | '
                        'valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                            epoch, (time.time() - epoch_start_time),
                            val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            # if not best_val_loss or val_loss < best_val_loss:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val_loss = val_loss
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    # Load the best saved model.
    # with open(args.save, 'rb') as f:
    #     model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, corpus, test_data,
                         criterion, eval_batch_size, num_steps)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info(logger.info * 89)


if __name__ == '__main__':
    main()
