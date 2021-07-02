# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 14:51
"""
import torch
import torch.nn as nn


class RNNEncoder(nn.Module):

    def __init__(self, input_size, rnn_type, hidden_size, bidirectional, num_layers, dropout):
        super().__init__()

        self.input_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=input_size, bidirectional=bidirectional, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        output, hidden = self.rnn(self.input_dropout(input))

        return output, hidden

