import torch
from torch import nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        #self.embedding = nn.Embedding(input_dim, embedding_dim)
        #self.rnn = nn.LSTM(embedding_dim,
        #                   hidden_dim,
        #                   num_layers=n_layers,
        #                   bidirectional=bidirectional,
        #                   dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)),
                Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)))

    def forward(self, waveform):
        self.hidden = self.init_hidden()
        #x = self.embedding(waveform)
        x = waveform
        x, self.hidden = self.rnn(x, self.hidden)
        hidden, cell = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x = self.fc(hidden)

        return x
