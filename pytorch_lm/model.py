import torch
import torch.nn as nn

from pytorch_lm.dropout import create_dropout
from pytorch_lm.utils.config import create_object
from pytorch_lm.utils.lang import public_dict
from pytorch_lm.weight_drop import WeightDrop


class LMModel(nn.Module):
    """Base model for (RNN-based) language models."""
    def init_hidden(self, batch_size):
        """Initializes the hidden state."""
        raise NotImplementedError(
            'init_hidden is not implemented by class {}'.format(
                self.__class__.__name__))

    def loss_regularizer(self):
        """
        The regularizing term (if any) that is added to the loss. This function
        may be stateful, i.e. the model might store references to the current
        batch in its inner states to compute the regularizing term.
        """
        return 0


class GenericRnnModel(LMModel):
    """
    Implements a generic embedding - RNN - softmax LM.

    Arguments:
    - vocab_size: vocab size
    - embedding_size: the size of the embedding (and the softmax). If either
                      this or the next parameter is 0, they are taken to be
                      equal
    - hidden_size: the number of units in the hidden LSTM layers. See above
    - rnn: a dictionary:
    {
      "class": the RNN subclass
      "args": its arguments (apart from input & hidden size and dropout prob.)
      "kwargs": its keyword arguments (likewise)
    }
    - embedding_dropout: per-row dropout on the embedding matrix
      (a dropout string)
    - output_dropout: the dropout applied on the RNN output
    - dropout: input + layer + output dropout. The individual parameters take
               precedence.
    """
    def __init__(self, vocab_size, num_layers, rnn=None,
                 embedding_size=0, hidden_size=0,
                 embedding_dropout=None, input_dropout=None,
                 layer_dropout=None, output_dropout=None,
                 dropout=None, weight_tying=False):
        super(GenericRnnModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size if hidden_size else embedding_size
        self.embedding_size = embedding_size if embedding_size else self.hidden_size
        if not (self.hidden_size or self.embedding_size):
            raise ValueError('embedding_size and hidden_size cannot be both 0')

        input_dropout = input_dropout or dropout
        layer_dropout = layer_dropout or dropout
        output_dropout = output_dropout or dropout
        # Embedding & output dropouts
        self.emb_do = nn.Dropout(float(embedding_dropout))
        self.in_do = create_dropout(input_dropout)
        self.lay_do = nn.ModuleList(
            [create_dropout(layer_dropout) for _ in range(num_layers - 1)])
        self.out_do = create_dropout(output_dropout)

        self.encoder = nn.Embedding(vocab_size, self.embedding_size)
        if not rnn:
            rnn = {'class': 'DefaultLstmLayer'}
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.embedding_size if not l else self.hidden_size
            out_size = self.hidden_size if l + 1 != num_layers else self.embedding_size
            self.layers.append(
                create_object(rnn, base_module='pytorch_lm.rnn',
                              args=[in_size, out_size])
            )
        self.decoder = nn.Linear(self.embedding_size, vocab_size)

        if weight_tying:
            # Linear.weight is transposed, so this will just work
            self.decoder.weight = self.encoder.weight

    # ----- OK, I am not sure this is the best one can come up with, but -----
    # ----- I really want to keep PressAndWolfModel as a separate class

    def forward(self, input, hidden):
        emb = self._encode(input)
        output, hidden, outputs, raw_outputs = self._rnn(emb, hidden)
        decoded = self._decode(output)
        return decoded, hidden

    def _encode(self, input):
        """Encodes the input with the encoder."""
        emb = self.encoder(input)

        # Embedding dropout
        if self.emb_do and self.training:
            # Creates the input embedding mask. A much faster version of the
            # solution
            # from https://github.com/julian121266/RecurrentHighwayNetworks
            mask = self.emb_do(torch.ones_like(input).type_as(emb)).cpu()

            for batch in range(mask.size()[0]):
                m = {}
                for n1 in range(mask.size()[1]):
                    x = m.setdefault(input[batch][n1].item(), mask[batch][n1].item())
                    mask[batch][n1] = x
            # type_as() puts it mask back to cuda if emb is there
            emb = emb * mask.type_as(emb).unsqueeze(2).expand_as(emb)
        return emb

    def _rnn(self, emb, hidden):
        """Runs the RNN on the embedded input."""
        # self.rnn.flatten_parameters()
        raw_outputs = []
        outputs = []
        new_hidden = []

        input = self.in_do(emb)
        for l, rnn in enumerate(self.layers):
            l_output, l_hidden = rnn(input, hidden[l])
            raw_outputs.append(l_output)
            new_hidden.append(l_hidden)
            if l != self.num_layers - 1:
                output = self.lay_do[l](l_output)
                input = output
            else:
                output = self.out_do(l_output)
            outputs.append(output)

        # raw_outputs and outputs are returned so that activation regularization
        # (see Merity et al. 2018) can be done
        return output, new_hidden, raw_outputs, outputs

    def _decode(self, output):
        """Runs softmax (etc.) on the output of the (last) RNN layer."""
        output = output.contiguous()
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    # ----- End of forward() ------

    def init_hidden(self, batch_size):
        return [rnn.init_hidden(batch_size) for rnn in self.layers]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               dict(public_dict(self), rnn=self.layers))


class SmallLstmModel(GenericRnnModel):
    def __init__(self, vocab_size):
        super(SmallLstmModel, self).__init__(vocab_size, 2, hidden_size=200)


class MediumLstmModel(GenericRnnModel):
    def __init__(self, vocab_size):
        super(MediumLstmModel, self).__init__(vocab_size, 2, hidden_size=650,
                                              dropout=0.5)


class LargeLstmModel(GenericRnnModel):
    def __init__(self, vocab_size):
        super(LargeLstmModel, self).__init__(vocab_size, 2, hidden_size=1500,
                                             dropout=0.65)


class PressAndWolfModel(GenericRnnModel):
    """
    Optionally tie weights as in:
    "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    https://arxiv.org/abs/1608.05859
    and
    "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
    https://arxiv.org/abs/1611.01462

    The default is weight tying and projection with lambda = 0.15, as in the
    paper.
    """
    def __init__(self, vocab_size, num_layers, embedding_size=0, hidden_size=0,
                 rnn=None, embedding_dropout=None, output_dropout=None,
                 dropout=None, weight_tying=True, projection_lambda=0.15):
        super(PressAndWolfModel, self).__init__(
            vocab_size, num_layers, embedding_size, hidden_size, rnn,
            embedding_dropout, output_dropout, dropout, weight_tying
        )

        if projection_lambda:
            self.projection = nn.Linear(embedding_size, embedding_size, bias=False)
            self.projection_lambda = projection_lambda
        else:
            self.projection = None

    def _rnn(self, emb, hidden):
        """Also performs the projection."""
        output, hidden = super(PressAndWolfModel, self)._rnn(emb, hidden)
        return self.projection(output) if self.projection else output

    def loss_regularizer(self):
        """The regularizing term (if any) that is added to the loss."""
        if self.projection:
            return self.projection.weight.norm() * self.projection_lambda
        else:
            return 0


class SmallPressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(SmallPressAndWolfModel, self).__init__(
            vocab_size, 2, hidden_size=200)


class MediumPressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(MediumPressAndWolfModel, self).__init__(
            vocab_size, 2, hidden_size=650, dropout=0.5)


class LargePressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(LargePressAndWolfModel, self).__init__(
            vocab_size, 2, hidden_size=1500, dropout=0.65)


class MerityModel(GenericRnnModel):
    """
    Most of tricks in "Regularizing and Optimizing LSTM Language Models"
    (Merity et al. 2018) have been added to the generic models. However,
    AR and TAR do warrant another model class.
    """
    def __init__(self, vocab_size, embedding_size=0, hidden_size=0,
                 rnn=None, embedding_dropout=None, output_dropout=None,
                 weight_tying=True, alpha=0, beta=0):
        """The new parameters are alpha for AR and beta for TAR."""
        super(MerityModel, self).__init__(
            vocab_size, embedding_size, hidden_size, rnn,
            embedding_dropout, output_dropout, weight_tying
        )
        self.alpha = alpha
        self.beta = beta
        self.loss_reg = 0

    def _rnn(self, emb, hidden):
        """
        Runs the RNN on the embedded input. Also computes the regularization
        loss (both AR and TAR are activation regularizers, so they can only be
        computed while the data tensors are available.
        """
        raw_output, hidden = self.rnn(emb, hidden)
        self.out_do.reset_noise()
        output = torch.stack([self.out_do(raw_o) for raw_o in raw_output], 1)
        raw_output = torch.stack(raw_output, 1)
        self.loss_reg = 0
        if self.beta:
            self.loss_reg += self.beta * (
                raw_output[:, 1:] - raw_output[:, :-1]).pow(2).mean()
        if self.alpha:
            self.loss_reg += self.alpha * output.pow(2).mean()
        return output, hidden

    def loss_regularizer(self):
        """AR + TAR."""
        return self.loss_reg


class RealMerityModel(GenericRnnModel):
    """
    Imported and converted from the original repository.
    """
    def __init__(self, vocab_size, num_layers, rnn=None,
                 embedding_size=0, hidden_size=0,
                 embedding_dropout=None, input_dropout=None,
                 layer_dropout=None, output_dropout=None,
                 dropout=None, weight_tying=True,
                 dropconnect=0, alpha=0, beta=0):
        """The new parameters are alpha for AR and beta for TAR."""
        super(RealMerityModel, self).__init__(
            vocab_size, num_layers, {'class': 'PytorchLstmLayer'},
            embedding_size, hidden_size, embedding_dropout, input_dropout,
            layer_dropout, output_dropout, dropout, weight_tying
        )
        self.dropconnect = dropconnect

        layers = [WeightDrop(rnn, ['weight_hh_l0'], dropout=dropconnect)
                  for rnn in self.layers]
        del self.layers[:]
        for rnn in layers:
            self.layers.append(rnn)

        self.alpha = alpha
        self.beta = beta
        self.loss_reg = 0

    def _rnn(self, emb, hidden):
        """
        Runs the RNN on the embedded input. Also computes the regularization
        loss (both AR and TAR are activation regularizers, so they can only be
        computed while the data tensors are available.
        """
        output, hidden, raw_outputs, outputs = super(RealMerityModel, self)._rnn(
            emb, hidden)

        self.loss_reg = 0
        raw_output = raw_outputs[-1]
        if self.beta:
            self.loss_reg += self.beta * (
                raw_output[:, 1:] - raw_output[:, :-1]).pow(2).mean()
        if self.alpha:
            self.loss_reg += self.alpha * output.pow(2).mean()
        return output, hidden, raw_outputs, outputs

    def loss_regularizer(self):
        """AR + TAR."""
        return self.loss_reg
