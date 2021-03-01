import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channel, stride, kernel_size=4, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class TransposeConv2d(nn.Module):
    def __init__(self, in_channels, out_channel, is_tanh=False):
        super().__init__()

        if is_tanh:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channel, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class Conv2dInstance(nn.Module):
    def __init__(self, in_channels, conv_dim):
        super(Conv2dInstance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, 4, 2, 1),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channel, stride, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3):
        super(DenseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_layer=3, growth_rate=32):
        super(ResidualBlock, self).__init__()
        denses = []
        for i in range(n_layer):
            denses.append(DenseConv(in_channels + growth_rate * i, growth_rate))
        self.denses = nn.Sequential(*denses)

        self.lff = nn.Conv2d(in_channels + growth_rate * n_layer, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.denses(x))


class GlobalResidual(nn.Module):
    def __init__(self, in_channel, n_channel, n_blocks=3, growth_rate=32):
        super(GlobalResidual, self).__init__()
        self.sfe1 = nn.Conv2d(in_channel, n_channel, kernel_size=3, padding=3 // 2)

        self.resBlocks = nn.ModuleList()
        for i in range(n_blocks):
            self.resBlocks.append(ResidualBlock(n_channel, growth_rate=growth_rate))

        self.gff = nn.Sequential(
            nn.Conv2d(n_channel * n_blocks, n_channel, kernel_size=1),
            nn.Conv2d(n_channel, in_channel, kernel_size=3, padding=3 // 2)
        )

    def forward(self, x):
        h = self.sfe1(x)

        local_features = []
        for i in range(len(self.resBlocks)):
            h = self.resBlocks[i](h)
            local_features.append(h)

        h = self.gff(torch.cat(local_features, 1)) + x
        return h
