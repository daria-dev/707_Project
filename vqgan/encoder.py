from numpy import append
import torch
from util import ResidualBlock
from util import DownSample
from util import NonLocalBlock
from util import GroupNorm
from util import Swish

'''
    VQGAN Encoder
    - Architecture as in paper
    - input: H x W x input_channels
    - output: h x w x latent_space_dim
'''
class Encoder(torch.nn.Module):
    def __init__(self, input_channels, latent_space_dim):
        super(Encoder, self).__init__()
        chanel_dim = [128, 128, 256, 256, 512]

        # initial conv
        layers = [torch.nn.Conv2d(input_channels, chanel_dim[0], 3, 1, 1)]

        # donwample steps
        for i in range(len(chanel_dim) - 1):
            in_channels = chanel_dim[i]
            out_channels = chanel_dim[i+1]

            layers.append(ResidualBlock(in_channels, out_channels))
            layers.append(ResidualBlock(out_channels, out_channels))
            layers.append(DownSample(out_channels))

        layers.append(ResidualBlock(chanel_dim[-1], chanel_dim[-1]))
        layers.append(NonLocalBlock(chanel_dim[-1]))
        layers.append(ResidualBlock(chanel_dim[-1], chanel_dim[-1]))
        layers.append(GroupNorm(chanel_dim[-1]))
        layers.append(Swish())

        layers.append(torch.nn.Conv2d(chanel_dim[-1], latent_space_dim, 3, 1, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)