#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Taken from the word_language_model directory of the pytorch/examples repository.
"""

import os
import random

import numpy as np
import torch
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """Loads the whole training (i.e. inc. valid, eval) corpus into memory."""
    def __init__(self, path, shuffle_train=True):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), shuffle_train)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), False)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), False)

    def tokenize(self, path, shuffle):
        """Tokenizes a text file. Optionally shuffles the sentences."""
        assert os.path.exists(path)
        # Add words to the dictionary
        text = []
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                text.append([self.dictionary.add_word(word) for word in words])
        if shuffle:
            random.shuffle(text)

        return np.array([wid for sentence in text for wid in sentence])


class LMData(object):
    """
    Takes a Corpus and creates (batched) LM training data from it. This class
    is Pytorch-specific.
    """
    def __init__(self, text, batch_size, cuda):
        """
        Batchifies text.

        Arguments:
        - text: text which has already been tokenized, and words
                converted to int ids
        - batch_size: the batch size
        - cuda: whether the tensors should be created on the GPU or not
        """
        self.batch_size = batch_size
        self.cuda = cuda
        self.data = self.batchify(text)

    def batchify(self, text):
        """
        Starting from sequential data, batchify arranges the dataset into rows.
        For instance, with the alphabet as the sequence and batch size 4, we'd
        get
        ┌ a b c d e f ┐
        │ g h i j k l │
        │ m n o p q r │
        │ s t u v w x ┘.
        These rows are treated as independent by the model, which means that
        the dependence of e. g. 'g' on 'f' can not be learned, but allows more
        efficient batch processing.
        """
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = len(text) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = text[:nbatch * self.batch_size]
        # Evenly divide the data across the bsz batches.
        data = data.reshape(self.batch_size, -1)
        data = torch.from_numpy(data).long().contiguous()
        if self.cuda:
            data = data.cuda()
        return data

    def _get_batches(self, num_steps, evaluation):
        """
        Does the actual splitting of the data into minibatches. Called by
        get_batches().
        """
        # -1, because we need at least 2 items (input, output)
        for i in range(0, self.data.size(1) - 1, num_steps):
            seq_len = min(num_steps, self.data.size(1) - 1 - i)
            data_chunk = self.data[:, i:i+seq_len].contiguous()
            target_chunk = self.data[:, i+1:i+1+seq_len].contiguous()  # .view(-1))
            yield data_chunk, target_chunk

    def get_batches(self, num_steps, evaluation=False):
        """
        get_minibatch iterates through the batchified data, returning a chunk of
        length num_steps. Continuing the example from the batchify method, with
        num_steps = 3, the following two Variables are returned on the first
        call:
        ┌ a b c ┐ ┌ b c d ┐
        │ g h i │ │ h i j │
        │ g n o │ │ n o p │
        └ s t u ┘ └ t u v ┘
        The chunks are along dimension 1, corresponding to the seq_len
        dimension in the Lstm class, but unlike the LSTM in Pytorch.

        Returns the two tensors above, and the learning rate ratio if
        ``evaluate`` is ``False``.

        Arguments:
        - num_steps: the BPTT sequencer object
        - evaluation: whether the minibatch will be used in evaluation (i.e. it
                      doesn't need gradients) or not
        """
        seq_len, lr_ratio = num_steps.num_steps()
        for data_chunk, target_chunk in self._get_batches(seq_len, evaluation):
            # TODO can we no_grad target as well?
            if evaluation:
                with torch.no_grad():
                    data = Variable(data_chunk)
            else:
                data = Variable(data_chunk)
            target = Variable(target_chunk)  # .view(-1))
            if not evaluation:
                yield data, target, lr_ratio
            else:
                yield data, target
