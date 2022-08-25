import torch
import torch.nn as nn
from torch.autograd import Variable



class MLP(nn.Module):
    def __init__(self,input_size,time_seq,hidden_dim,dropout=0.5):
        super(MLP, self).__init__()
        self.input_size = input_size
        input_area = input_size[0] * input_size[1]
        # self.time_seq = time_seq
        dim = input_size[0] * input_size[1] * time_seq
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim[1], input_area)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        batchsize = x.size(0)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = x.reshape(batchsize,1,1,self.input_size[0],self.input_size[1])
        #print(x.shape)
        return x