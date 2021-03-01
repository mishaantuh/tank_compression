import numpy as np
import torch.nn as nn

from .layers import Conv2d, TransposeConv2d, GlobalResidual


class Generator(nn.Module):
    def __init__(self, n_channel=64, n_layers=6):
        super(Generator, self).__init__()

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
        dec_in_channel = n_channel
        for i in range(n_layers):
            if i + 1 == n_layers:
                layers_dec.append(TransposeConv2d(dec_in_channel, 3, is_tanh=True))
                continue

            layers_dec.append(TransposeConv2d(dec_in_channel, n_channel // (2 ** (i + 1))))
            dec_in_channel = n_channel // (2 ** (i + 1))
        self.dec = nn.Sequential(*layers_dec)

    def forward(self, x, mode="gen"):
        if mode == "encode":
            return self.enc(x)
        if mode == "decode":
            return self.dec(x)
        else:
            z = self.enc(x)
            return self.dec(z)
