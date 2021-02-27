import torch
import numpy as np
import torch.nn as nn

from .layers import Conv2d, TransposeConv2d, GlobalResidual

class Generator(nn.Module):
    def __init__(self, n_channel=64, n_layers=5, n_attrs=9):
        super(Generator, self).__init__()
        self.n_attrs = n_attrs

        # <--- encoder ---> #
        layers_enc = []
        enc_in_channel = 3
        for i in range(n_layers):
            layers_enc.append(Conv2d(enc_in_channel, n_channel * 2 ** i, 2, 4))
            enc_in_channel = n_channel * 2 ** i
        self.enc = nn.Sequential(*layers_enc)

        # <--- decoder ---> #
        layers_dec = []
        n_channel = 2 ** (int(np.log2(n_channel)) + n_layers - 1)
        dec_in_channel = n_channel + n_attrs
        for i in range(n_layers):
            if i >= n_layers - 1:
                self.layers_dec.append(GlobalResidual(dec_in_channel, dec_in_channel * 2))
            if i + 1 == n_layers:
                layers_dec.append(TransposeConv2d(dec_in_channel, 3, is_tanh=True))
                continue

            layers_dec.append(TransposeConv2d(dec_in_channel, n_channel // (2 ** (i + 1))))
            dec_in_channel = n_channel // (2 ** (i + 1))
        self.dec = nn.Sequential(*layers_dec)

    def forward(self, x, attr=None, mode="gen"):
        if mode == "encode":
            return self.enc(x)
        if mode == "decode":
            return self.dec(x)
        else:
            z = self.enc(x)
            return self.dec(z)
