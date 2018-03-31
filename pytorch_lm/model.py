import torch
import torch.nn as nn

from pytorch_lm.lstm import Lstm
from pytorch_lm.dropout import create_dropout


class LMModel(nn.Module):
    """Base model for (RNN-based) language models."""
    def init_hidden(self, batch_size):
        """Initializes the hidden state."""
        raise NotImplementedError(
            'init_hidden is not implemented by class {}'.format(
                self.__class__.__name__))

    def loss_regularizer(self):
        """The regularizing term (if any) that is added to the loss."""
        return 0


class GenericLstmModel(LMModel):
    """
    Implements a generic embedding - LSTM - softmax LM.

    The only interesting parameter here is cell_data, which is a dictionary:
    {
      "class": the LstmCell subclass
      "args": its arguments (apart from input & hidden size and dropout prob.)
      "kwargs": its keyword arguments (likewise)
    }
    """
    def __init__(self, vocab_size, hidden_size=200, num_layers=2, dropout=0.5,
                 cell_data=None, embedding_dropout=None, output_dropout=None):
        super(GenericLstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding & output dropouts
        self.emb_do = create_dropout(embedding_dropout, True)
        self.out_do = create_dropout(output_dropout)

        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.rnn = Lstm(hidden_size, hidden_size, num_layers, dropout,
                        cell_data)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    # ----- OK, I am not sure this is the best one can come up with, but -----
    # ----- I really want to keep PressAndWolfModel as a separate class

    def forward(self, input, hidden):
        emb = self._encode(input)
        output, hidden = self._rnn(emb, hidden)
        decoded = self._decode(output)
        return decoded, hidden

    def _encode(self, input):
        """Encodes the input with the encoder."""
        emb = self.encoder(input)

        # Embedding dropout
        if self.emb_do:
            self.emb_do.reset_noise()
            mask = self.emb_do(torch.ones_like(input).type_as(emb))
            emb = emb * mask.unsqueeze(2).expand_as(emb)
        return emb

    def _rnn(self, emb, hidden):
        """Runs the RNN on the embedded input."""
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        self.out_do.reset_noise()
        output = self.out_do(output)
        return output, hidden

    def _decode(self, output):
        """Runs softmax (etc.) on the output of the (last) RNN layer."""
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    # ----- End of forward() ------

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)


class SmallLstmModel(GenericLstmModel):
    def __init__(self, vocab_size):
        super(SmallLstmModel, self).__init__(vocab_size, 200, 2, 0)


class MediumLstmModel(GenericLstmModel):
    def __init__(self, vocab_size):
        super(MediumLstmModel, self).__init__(vocab_size, 650, 2, 0.5,
                                              output_dropout='0.5')


class LargeLstmModel(GenericLstmModel):
    def __init__(self, vocab_size):
        super(LargeLstmModel, self).__init__(vocab_size, 1500, 2, 0.65,
                                             output_dropout='0.65')


class PressAndWolfModel(GenericLstmModel):
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
    def __init__(self, vocab_size, hidden_size=200, num_layers=2, dropout=0.5,
                 cell_data=None, embedding_dropout=None, output_dropout=None,
                 projection_lambda=0.15, weight_tying=True):
        super(PressAndWolfModel, self).__init__(
            vocab_size, hidden_size, num_layers, dropout,
            cell_data, embedding_dropout)

        if weight_tying:
            # Linear.weight is transposed, so this will just work
            self.decoder.weight = self.encoder.weight
        if projection_lambda:
            self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.projection_lambda = projection_lambda
        else:
            self.projection = None

    def _rnn(self, emb, hidden):
        """Also performs the projection."""
        output = super(PressAndWolfModel, self)._rnn(emb, hidden)
        return self.projection(output) if self.projection else output

    def loss_regularizer(self):
        """The regularizing term (if any) that is added to the loss."""
        if self.projection:
            return self.projection.weight.norm() * self.projection_lambda
        else:
            return 0


class SmallPressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(SmallPressAndWolfModel, self).__init__(vocab_size, 200, 2, 0)


class MediumPressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(MediumPressAndWolfModel, self).__init__(vocab_size, 650, 2, 0.5)


class LargePressAndWolfModel(PressAndWolfModel):
    def __init__(self, vocab_size):
        super(LargePressAndWolfModel, self).__init__(vocab_size, 1500, 2, 0.65)
