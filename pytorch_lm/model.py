import torch
import torch.nn as nn

from pytorch_lm.lstm_pytorch import Lstm


class SmallZarembaModel(nn.Module):
    """"Implements the small model from Zaremba (2014)."""
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__()
        self.hidden_size = 200
        self.input_size = 200
        self.num_layers = 2

        self.encoder = nn.Embedding(vocab_size, self.hidden_size)
        self.rnn = Lstm(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
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

    def save_parameters(self, out_dict=None, prefix=''):
        if out_dict is None:
            out_dict = {}
        self.rnn.save_parameters(out_dict, prefix=prefix + 'RNN/')
        out_dict[prefix + 'embedding'] = self.encoder.weight.data.cpu().numpy()
        # .T is required because stupid Linear stores the weights transposed
        out_dict[prefix + 'softmax_w'] = self.decoder.weight.data.cpu().numpy().T
        out_dict[prefix + 'softmax_b'] = self.decoder.bias.data.cpu().numpy()
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        def set_data(parameter, value, is_cuda):
            t = torch.from_numpy(value)
            if is_cuda:
                t = t.cuda()
            parameter.data = t

        is_cuda = self.encoder.weight.is_cuda
        self.rnn.load_parameters(data_dict, prefix=prefix + 'RNN/')
        set_data(self.encoder.weight, data_dict[prefix + 'embedding'], is_cuda)
        # .T is required because stupid Linear stores the weights transposed
        set_data(self.decoder.weight, data_dict[prefix + 'softmax_w'].T, is_cuda)
        set_data(self.decoder.bias, data_dict[prefix + 'softmax_b'], is_cuda)
