import torch.nn as nn

from pytorch_lm.lstm import Lstm


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


class CustomZarembaModel(LMModel):
    """Implements a generic embedding - LSTM - softmax LM."""
    def __init__(self, vocab_size, hidden_size=200, num_layers=2, dropout=0.5):
        super(CustomZarembaModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.rnn = Lstm(hidden_size, hidden_size, num_layers, dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)


class SmallZarembaModel(CustomZarembaModel):
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__(vocab_size, 200, 2, 0)


class MediumZarembaModel(CustomZarembaModel):
    def __init__(self, vocab_size):
        super(MediumZarembaModel, self).__init__(vocab_size, 650, 2, 0.5)


class LargeZarembaModel(CustomZarembaModel):
    def __init__(self, vocab_size):
        super(LargeZarembaModel, self).__init__(vocab_size, 1500, 2, 0.65)


class PressAndWolfModel(CustomZarembaModel):
    """
    Optionally tie weights as in:
    "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    https://arxiv.org/abs/1608.05859
    and
    "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
    https://arxiv.org/abs/1611.01462

    The default is no weight tying and no projection -- i.e. it defaults to
    the usual Zaremba model.
    """
    def __init__(self, vocab_size, hidden_size=200, num_layers=2, dropout=0.5,
                 projection_lambda=0, weight_tying=False):
        super(PressAndWolfModel, self).__init__(
            vocab_size, hidden_size, num_layers, dropout)
        if weight_tying:
            # Linear.weight is transposed, so this will just work
            self.decoder.weight = self.encoder.weight
        if projection_lambda:
            self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.projection_lambda = projection_lambda
        else:
            self.projection = None

    def forward(self, input, hidden):
        emb = self.encoder(input)
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        projected = self.projection(output) if self.projection else output
        decoded = self.decoder(
            projected.view(projected.size(0) * projected.size(1),
                           projected.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

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
