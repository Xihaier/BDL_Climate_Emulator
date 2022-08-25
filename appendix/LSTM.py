import torch
import torch.nn as nn
from torch.autograd import Variable

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_size, time_seq,hidden_dim,
                 num_layers=2,batch_first=True,pca=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.input_dim = input_size[0] * input_size[1]
        self.hidden_dim = hidden_dim
        self.time_seq = time_seq
        # self.batch_size = batch_size
        self.num_layers = num_layers
        self.pca = pca
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        output_dim = self.input_dim
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

        self.batch_first = batch_first

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        batch_size = input.size(0)
        tem_seq = input.size(1)
        if self.batch_first:
            if self.pca:
                x = input.permute(1,0,2,3)
            else:
                x = input.permute(1, 0, 2, 3, 4)
        x = x.view(self.time_seq,batch_size,-1)
        lstm_out, self.hidden = self.lstm(x)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(batch_size, -1))

        y_pred = y_pred.reshape(batch_size, 1, 1, self.input_size[0], self.input_size[1])

        return y_pred