#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements a very basic version of LSTM."""

import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_lm.dropout import StatefulDropout, create_dropout
from pytorch_lm.utils.config import create_object
from pytorch_lm.utils.lang import public_dict


class InitHidden(object):
    """
    Provides init_hidden to subclasses. Requires that self.hidden size is
    available.
    """
    def init_hidden(self, batch_size):
        """Returns the Variables for the hidden states."""
        weight = next(self.parameters()).data
        if self.batch_first:
            dims = (batch_size, 1, self.hidden_size)
        else:
            dims = (1, batch_size, self.hidden_size)
        return (Variable(weight.new(dims).zero_()),
                Variable(weight.new(dims).zero_()))


class LstmLayer(nn.Module, InitHidden):
    """
    A reimplementation of the LSTM cell. (Actually, a layer of LSTM cells.)

    As a reminder: input size is seq_len x batch_size x input_features,
    following the PyTorch convention; but also see the batch_first parameter.
    """
    def __init__(self, input_size, hidden_size, batch_first=False,
                 forget_bias=1):
        """
        Args:
            - input_size: the number of input features
            - hidden_size: the number of cells
            - batch_first: if True, the first two dimensions are swapped [False]
            - forget_bias: the value of the forget bias [1]
        """
        super(LstmLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.w_i = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.w_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.Tensor(4 * hidden_size))

        # f_bias is deleted by the pre-format hook so that it runs only once
        # The forget gate is initialized before the first forward call,
        # because the weights are initialized after __init__()
        self.f_bias = self.forget_bias = forget_bias
        self.register_forward_pre_hook(LstmLayer.initialize_f)

    @classmethod
    def initialize_f(cls, module, _):
        """Initializes the forget gate."""
        if module.f_bias is not None:
            _, b_f, _, _ = module.b.data.chunk(4, 0)
            b_f.fill_(module.f_bias)
            module.f_bias = None

    def forward(self, inputs, hidden):
        """
        Runs the layer on the sequence inputs. Calls forward_one in a loop.
        """
        outputs = []
        seq_dim = 1 if self.batch_first else 0
        h_t, c_t = map(h.squeeze(seq_dim) for h in hidden)

        # chunk() cuts batch_size x 1 x input_size chunks from input
        for input_t in input.chunk(input.size(1), dim=seq_dim):
            values = input_t.squeeze(seq_dim)  # From input to output
            h_t, c_t = self.forward_one(values, (h_t, c_t))
            values = h_t
            outputs.append(values)

        return (torch.stack(outputs, seq_dim),
                (h_t.unsqueeze(seq_dim), c_t.unsqueeze(seq_dim)))

    def forward_one(self, input, hidden):
        """Of course, forward must be implemented in subclasses."""
        raise NotImplementedError()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, public_dict(self))


class PytorchLstmLayer(nn.LSTM, InitHidden):
    """Wraps the PyTorch LSTM object."""
    pass


class DefaultLstmLayer(LstmLayer):
    """
    The default LSTM implementation. No dropout, as that is handled outside.
    """
    def forward_one(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.w_h)
        ifgo += self.b

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class DropoutLstmLayer(LstmLayer):
    """An LstmLayer with H->H dropout."""
    def __init__(self, input_size, hidden_size, batch_first=False,
                 forget_bias=1, hh_dropout=0):
        super(DropoutLstmLayer, self).__init__(
            input_size, hidden_size, batch_first, forget_bias)
        self.hh_dropout = hh_dropout

        self.do = self.create_dropouts()
        for i, do in enumerate(self.do):
            self.add_module('do{}'.format(i), do)

    def create_dropouts(self):
        """
        Creates the ``list`` of :class:`Dropout` objects used by the cell.
        This is one method to be implemented; this default implementation
        returns a single Dropout object created with create_dropout().
        """
        return [create_dropout(self.dropout)]

    def forward(self, inputs, hidden):
        """
        Runs the layer on the sequence inputs. Initializes the :class:`Dropout`
        objects and calls forward_one in a loop.
        """
        for do in self.do:
            do.reset_noise()

        return super(DropoutLstmLayer, self).forward(inputs, hidden)


class MoonLstmLayer(DropoutLstmLayer):
    """
    Following Moon et al. (2015), dropout (with a per-sequence mask) is applied
    on c_t. Note: this sucks.
    """
    def forward_one(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.w_h)
        ifgo += self.b

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = self.do[0](f * c_t + i * g)
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class TiedGalLstmLayer(LstmLayer):
    """
    Following Gal and Ghahramani (2016), per-sequence dropout is applied on
    both the input and h_t. Also known as VD-LSTM. With tied gates.
    """
    def forward(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + self.do[0](h_t).matmul(self.w_h)
        ifgo += self.b

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class UntiedGalLstmLayer(LstmLayer):
    """
    Following Gal and Ghahramani (2016), per-sequence dropout is applied on
    both the input and h_t. Also known as VD-LSTM. With untied weights.
    """
    def create_dropouts(self):
        return [StatefulDropout(self.dropout)
                for _ in range(4)]

    def forward(self, input, hidden):
        h_t, c_t = hidden

        w_ii, w_if, w_ig, w_io = self.w_i.chunk(4, 1)
        w_hi, w_hf, w_hg, w_ho = self.w_h.chunk(4, 1)
        b_i, b_f, b_g, b_o = self.b.chunk(4, 0)

        i = torch.sigmoid(input.matmul(w_ii) +
                          self.do[0](h_t).matmul(w_hi) + b_i)
        f = torch.sigmoid(input.matmul(w_if) +
                          self.do[1](h_t).matmul(w_hf) + b_f)
        g = torch.tanh(input.matmul(w_ig) +
                       self.do[2](h_t).matmul(w_hg) + b_g)
        o = torch.sigmoid(input.matmul(w_io) +
                          self.do[3](h_t).matmul(w_ho) + b_o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class SemeniutaLstmLayer(LstmLayer):
    """Following Semeniuta et al. (2016), dropout is applied on g_t."""
    def forward(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.w_h)
        ifgo += self.b

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * self.do[0](g)
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class MerityLstmLayer(LstmLayer):
    """
    Following Merity et al. (2018): uses DropConnect instead of Dropout, on
    the hidden-to-hidden matrices. The parameter of the DropConnect probability
    is still called dropout, unfortunately.
    """
    def forward(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.do[0](self.w_h))
        ifgo += self.b

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class Lstm(nn.Module):
    """
    Several layers of LstmCells. Input is batch_size x num_steps x input_size,
    which is different from the Pytorch LSTM (the first two dimensions are
    swapped).

    If dropout is specified, it is applied on the output. So for L layers,
    dropout is applied L + 1 times (once on the input in each layer + once on
    the final output).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,
                 input_dropout=0, layer_dropout=0, cell_data=None):
        """
        Arguments:
        - input_size: the size of the input vector
        - hidden_size: the size of the hidden vector
        - num_layers: the number of layers
        - dropout: the dropout used before, and between, the cells (output
                   dropout is handled in the model). The
                   individual parameters (see below) take precedence: if one of
                   them is not 0, the "umbrella" dropout argument is ignored
        - input_dropout: the dropout used on the input of the first layer
        - layer_dropout: the dropout used between layers
        - cell_data: the type of cell to use, with its __init__ arguments
        """
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if input_dropout or layer_dropout:
            self.input_dropout = input_dropout
            self.layer_dropout = layer_dropout
        else:
            self.input_dropout = dropout
            self.layer_dropout = dropout

        if not cell_data:
            cell_data = {'class': 'ZarembaLstmCell', 'args': [], 'kwargs': {}}

        self.layers = []
        for l in range(num_layers):
            in_size = input_size if not l else hidden_size
            out_size = hidden_size if l + 1 != num_layers else input_size
            do = self.input_dropout if not l else self.layer_dropout
            self.layers.append(
                create_object(
                    cell_data, base_module='pytorch_lm.rnn',
                    args=[in_size, out_size, do]
                )
            )
        for l, layer in enumerate(self.layers):
            self.add_module('Layer_{}'.format(l), layer)

    def forward(self, input, hiddens):
        outputs = []

        # To initialize per-sequence dropout
        for l in self.layers:
            for do in l.do:
                do.reset_noise()

        # chunk() cuts batch_size x 1 x input_size chunks from input
        for input_t in input.chunk(input.size(1), dim=1):
            values = input_t.squeeze(1)  # From input to output
            for l in range(self.num_layers):
                h_t, c_t = self.layers[l](values, hiddens[l])
                values = h_t
                hiddens[l] = h_t, c_t
            outputs.append(values)
        # outputs = torch.stack(outputs, 1)
        return outputs, hiddens

    def init_hidden(self, batch_size):
        return [self.layers[l].init_hidden(batch_size)
                for l in range(self.num_layers)]

    def save_parameters(self, out_dict=None, prefix=''):
        """
        Saves the parameters into a dictionary that can later be e.g. savez'd.
        If prefix is specified, it is prepended to the names of the parameters,
        allowing for hierarchical saving / loading of parameters of a composite
        model.
        """
        if out_dict is None:
            out_dict = {}
        for l, layer in enumerate(self.layers):
            self.layers[l].save_parameters(
                out_dict, prefix + 'Layer_' + str(l) + '/')
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        """Loads the parameters saved by save_parameters()."""
        for l, layer in enumerate(self.layers):
            key = prefix + 'Layer_' + str(l) + '/'
            part_dict = {k: v for k, v in data_dict.items() if k.startswith(key)}
            layer.load_parameters(part_dict, key)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, public_dict(self))
