import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .blocks import *


class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if mode == "LSTM":

            self.rnn_cell_list.append(
                LSTMCell(self.input_size, self.hidden_size, self.bias)
            )
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(
                    LSTMCell(self.hidden_size, self.hidden_size, self.bias)
                )

        elif mode == "GRU":
            self.rnn_cell_list.append(
                GRUCell(self.input_size, self.hidden_size, self.bias)
            )
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(
                    GRUCell(self.hidden_size, self.hidden_size, self.bias)
                )

        elif mode == "RNN_TANH":
            self.rnn_cell_list.append(
                RNNCell(self.input_size, self.hidden_size, self.bias, "tanh")
            )
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(
                    RNNCell(self.hidden_size, self.hidden_size, self.bias, "tanh")
                )

        elif mode == "RNN_RELU":
            self.rnn_cell_list.append(
                RNNCell(self.input_size, self.hidden_size, self.bias, "relu")
            )
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(
                    RNNCell(self.hidden_size, self.hidden_size, self.bias, "relu")
                )
        else:
            raise ValueError("Invalid RNN mode selected.")

        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if torch.cuda.is_available():
            h0 = Variable(
                torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
            )
        else:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        if torch.cuda.is_available():
            hT = Variable(
                torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
            )
        else:
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
            if self.mode == "LSTM":
                hidden_forward.append((h0[layer, :, :], h0[layer, :, :]))
            else:
                hidden_forward.append(h0[layer, :, :])

        hidden_backward = list()
        for layer in range(self.num_layers):
            if self.mode == "LSTM":
                hidden_backward.append((hT[layer, :, :], hT[layer, :, :]))
            else:
                hidden_backward.append(hT[layer, :, :])

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):

                if self.mode == "LSTM":
                    # If LSTM
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            input[:, t, :],
                            (hidden_forward[layer][0], hidden_forward[layer][1]),
                        )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            input[:, -(t + 1), :],
                            (hidden_backward[layer][0], hidden_backward[layer][1]),
                        )
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            hidden_forward[layer - 1][0],
                            (hidden_forward[layer][0], hidden_forward[layer][1]),
                        )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            hidden_backward[layer - 1][0],
                            (hidden_backward[layer][0], hidden_backward[layer][1]),
                        )

                else:
                    # If RNN{_TANH/_RELU} / GRU
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            input[:, t, :], hidden_forward[layer]
                        )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            input[:, -(t + 1), :], hidden_backward[layer]
                        )
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            hidden_forward[layer - 1], hidden_forward[layer]
                        )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            hidden_backward[layer - 1], hidden_backward[layer]
                        )

                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

            if self.mode == "LSTM":

                outs.append(h_forward_l[0])
                outs_rev.append(h_back_l[0])

            else:
                outs.append(h_forward_l)
                outs_rev.append(h_back_l)
        out = torch.stack([x.squeeze() for x in outs])
        out_rev = torch.stack([x.squeeze() for x in outs_rev])
        try:
            out = torch.einsum("ijk->jik", out)
            out_rev = torch.einsum("ijk->jik", out_rev)
        except:
            pass
        # print(out.shape)
        out = out.contiguous().view(-1, self.hidden_size)
        out_rev = out_rev.contiguous().view(-1, self.hidden_size)
        out = torch.cat((out, out_rev), 1)
        out = self.fc(out)

        return out
