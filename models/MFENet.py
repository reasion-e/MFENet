import torch
import torch.nn as nn
import math


from layers.Net import Net

from layers.revin import RevIN



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)



        self.net = Net(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.transpose(1,2)

        x = self.net(x)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x