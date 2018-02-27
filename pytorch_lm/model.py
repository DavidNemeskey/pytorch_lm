import torch.nn as nn

from pytorch_lm.lstm import Lstm


class CustomZarembaModel(nn.Module):
    """Implements a generic embedding - LSTM - softmax LM."""
    def __init__(self, vocab_size, hidden_size=200, num_layers=2,
                 weight_scale=0.1, dropout=0.5):
        super(CustomZarembaModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_scale = weight_scale
        self.dropout = dropout

        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.rnn = Lstm(hidden_size, hidden_size, num_layers, dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights(weight_scale)

    def init_weights(self, initrange):
        """Weight are initialized to between - and + initrange uniformly."""
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.uniform_(-initrange, initrange)  # fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

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
        super(SmallZarembaModel, self).__init__(vocab_size, 200, 2, 0.1, 0)


class MediumZarembaModel(CustomZarembaModel):
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__(vocab_size, 650, 2, 0.05, 0.5)


class LargeZarembaModel(CustomZarembaModel):
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__(vocab_size, 1500, 2, 0.04, 0.65)
