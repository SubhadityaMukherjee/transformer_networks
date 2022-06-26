import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .blocks import *

class SimpleRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, bias, output_size, activation="tanh"
    ):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.activation = activation

        self.rnn_cell_list = nn.ModuleList()
        if self.activation in ["relu", "tanh"]:
            self.create_list()
        else:
            raise ValueError("Invalid activation.")

        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def create_list(self):
        self.rnn_cell_list.append(
                RNNCell(self.input_size, self.hidden_size, self.bias, self.activation)
            )
        self.rnn_cell_list.extend(
                [
                    RNNCell(self.hidden_size, self.hidden_size, self.bias, self.activation) for l in range(1, self.num_layers)
                ]
            )

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(
                    torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
                )
            else:
                h0 = Variable(
                    torch.zeros(self.num_layers, input.size(0), self.hidden_size)
                )
        else:
            h0 = hx

        outs = []
        hidden = [h0[layer, :, :] for layer in range(self.num_layers)]

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1], hidden[layer]
                    )
                hidden[layer] = hidden_l
#
            outs.append(hidden_l)

        out = torch.stack([x.squeeze() for x in outs])
        try:
            out = torch.einsum("ijk->jik", out)
        except:
            pass
        # print(out.shape)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)

        return out
