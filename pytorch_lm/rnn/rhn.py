#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements Recurrent Highway Networks from Zilly et al. (2017)."""

import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_lm.dropout import create_dropout


class RhnLinTCTied(nn.Module):
    """Implements Recurrent Highway Networks from Zilly et al. (2017)."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,
                 transform_bias=None):
        super(RhnLinTCTied, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.do_h = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        self.do_t = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        for letter, do_list in [('H', self.do_h), ('T', self.do_t)]:
            self.add_module('Do_{}_w'.format(letter), do_list[0])
            for l, do in enumerate(do_list[1:], 1):
                self.add_module('Do_{}_{}'.format(letter, l), do)

        self.w_h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_t = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.r_h = [nn.Linear(hidden_size, hidden_size)
                    for l in range(self.num_layers)]
        self.r_t = [nn.Linear(hidden_size, hidden_size)
                    for l in range(self.num_layers)]
        for letter, lst in [('H', self.r_h), ('T', self.r_t)]:
            for l, p in enumerate(lst, 1):
                self.add_module('Rb_{}_{}'.format(letter, l), p)

    def forward(self, input, s):
        outputs = []

        # To initialize per-sequence dropout
        for do in self.do_h + self.do_t:
            do.reset_noise()

        # chunk() cuts batch_size x 1 x input_size chunks from input
        for input_t in map(torch.squeeze, input.chunk(input.size(1), dim=1)):
            for l in range(self.num_layers):
                # The input is processed only by the first layer
                whx = self.do_h[0](input_t).matmul(self.w_h) if l == 0 else 0
                wtx = self.do_t[0](input_t).matmul(self.w_t) if l == 0 else 0

                # The gates (and the state)
                h = torch.tanh(whx + self.r_h[l](self.do_h[l + 1](s)))
                t = torch.sigmoid(wtx + self.r_t[l](self.do_t[l + 1](s)))

                # The new state
                s = (h - s) * t + s

            # Here the output is the current s
            outputs.append(s)
        return torch.stack(outputs, 1), s

    def init_hidden(self, batch_size):
        """
        Returns a :class:`Variable` for the hidden state. As I understand, we
        only need one of these (as opposed to LSTM).
        """
        return Variable(torch.Tensor(
            batch_size, self.hidden_size).zero_().type(self.w_h.type()))


class Rhn(nn.Module):
    """Implements Recurrent Highway Networks from Zilly et al. (2017)."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,
                 transform_bias=None):
        super(Rhn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.do_h = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        self.do_t = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        self.do_c = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        for letter, do_list in [('H', self.do_h), ('T', self.do_t), ('C', self.do_c)]:
            self.add_module('Do_{}_w'.format(letter), do_list[0])
            for l, do in enumerate(do_list[1:], 1):
                self.add_module('Do_{}_{}'.format(letter, l), do)

        self.w_h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_t = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.r_hw = [nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                     for l in range(self.num_layers)]
        self.r_tw = [nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                     for l in range(self.num_layers)]
        self.r_cw = [nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                     for l in range(self.num_layers)]
        self.r_hb = [nn.Parameter(torch.Tensor(hidden_size))
                     for l in range(self.num_layers)]
        self.r_tb = [nn.Parameter(torch.Tensor(hidden_size))
                     for l in range(self.num_layers)]
        self.r_cb = [nn.Parameter(torch.Tensor(hidden_size))
                     for l in range(self.num_layers)]
        for letter, lst in [('H', self.r_hw), ('T', self.r_tw), ('C', self.r_cw)]:
            for l, p in enumerate(lst, 1):
                self.register_parameter('R_{}_{}_weight'.format(letter, l), p)
        for letter, lst in [('H', self.r_hb), ('T', self.r_tb), ('C', self.r_cb)]:
            for l, p in enumerate(lst, 1):
                self.register_parameter('R_{}_{}_bias'.format(letter, l), p)

    def forward(self, input, s):
        outputs = []

        # To initialize per-sequence dropout
        for do in self.do_h + self.do_t + self.do_c:
            do.reset_noise()

        # chunk() cuts batch_size x 1 x input_size chunks from input
        for input_t in map(torch.squeeze, input.chunk(input.size(1), dim=1)):
            for l in range(self.num_layers):
                # The input is processed only by the first layer
                whx = self.do_h[0](input_t).matmul(self.w_h) if l == 0 else 0
                wtx = self.do_t[0](input_t).matmul(self.w_t) if l == 0 else 0
                wcx = self.do_c[0](input_t).matmul(self.w_c) if l == 0 else 0

                rhs = self.do_h[l + 1](s).matmul(self.r_hw[l]) + self.r_hb[l]
                rts = self.do_t[l + 1](s).matmul(self.r_tw[l]) + self.r_tb[l]
                rcs = self.do_c[l + 1](s).matmul(self.r_cw[l]) + self.r_cb[l]

                # The gates (and the state)
                h = torch.tanh(whx + rhs)
                t = torch.sigmoid(wtx + rts)
                c = torch.sigmoid(wcx + rcs)

                # The new state
                s = h * t + s * c

            # Here the output is the current s
            outputs.append(s)
        return torch.stack(outputs, 1), s

    def init_hidden(self, batch_size):
        """
        Returns a :class:`Variable` for the hidden state. As I understand, we
        only need one of these (as opposed to LSTM).
        """
        return Variable(torch.Tensor(
            batch_size, self.hidden_size).zero_().type(self.w_h.type()))


class RhnLin(nn.Module):
    """Implements Recurrent Highway Networks from Zilly et al. (2017)."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,
                 transform_bias=None):
        super(RhnLin, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.do_h = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        self.do_t = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        self.do_c = [create_dropout(dropout) for _ in range(self.num_layers + 1)]
        for letter, do_list in [('H', self.do_h), ('T', self.do_t), ('C', self.do_c)]:
            self.add_module('Do_{}_w'.format(letter), do_list[0])
            for l, do in enumerate(do_list[1:], 1):
                self.add_module('Do_{}_{}'.format(letter, l), do)

        self.w_h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_t = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.r_h = [nn.Linear(hidden_size, hidden_size)
                    for l in range(self.num_layers)]
        self.r_t = [nn.Linear(hidden_size, hidden_size)
                    for l in range(self.num_layers)]
        self.r_c = [nn.Linear(hidden_size, hidden_size)
                    for l in range(self.num_layers)]
        for letter, lst in [('H', self.r_h), ('T', self.r_t), ('C', self.r_c)]:
            for l, p in enumerate(lst, 1):
                self.add_module('Rb_{}_{}'.format(letter, l), p)

    def forward(self, input, s):
        outputs = []

        # To initialize per-sequence dropout
        for do in self.do_h + self.do_t + self.do_c:
            do.reset_noise()

        # chunk() cuts batch_size x 1 x input_size chunks from input
        for input_t in map(torch.squeeze, input.chunk(input.size(1), dim=1)):
            for l in range(self.num_layers):
                # The input is processed only by the first layer
                whx = self.do_h[0](input_t).matmul(self.w_h) if l == 0 else 0
                wtx = self.do_t[0](input_t).matmul(self.w_t) if l == 0 else 0
                wcx = self.do_c[0](input_t).matmul(self.w_c) if l == 0 else 0

                # The gates (and the state)
                h = torch.tanh(whx + self.r_h[l](self.do_h[l + 1](s)))
                t = torch.sigmoid(wtx + self.r_t[l](self.do_t[l + 1](s)))
                c = torch.sigmoid(wcx + self.r_c[l](self.do_c[l + 1](s)))

                # The new state
                s = h * t + s * c

            # Here the output is the current s
            outputs.append(s)
        return torch.stack(outputs, 1), s

    def init_hidden(self, batch_size):
        """
        Returns a :class:`Variable` for the hidden state. As I understand, we
        only need one of these (as opposed to LSTM).
        """
        return Variable(torch.Tensor(
            batch_size, self.hidden_size).zero_().type(self.w_h.type()))
