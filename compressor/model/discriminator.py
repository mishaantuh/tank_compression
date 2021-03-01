from torch import nn

from .layers import Conv2dInstance


class Discriminator(nn.Module):
    def __init__(self, image_size=128, fc_dim=1024, n_channel=64, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3

        for i in range(n_layers):
            layers.append(Conv2dInstance(in_channels, n_channel * 2 ** i))
            in_channels = n_channel * 2 ** i

        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2 ** n_layers

        self.fc_adv = nn.Sequential(
            nn.Linear(n_channel * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        adv = self.fc_adv(y)
        return adv
