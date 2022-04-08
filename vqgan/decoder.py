import torch
from util import ResidualBlock
from util import UpSample
from util import NonLocalBlock
from util import GroupNorm
from util import Swish

'''
    VQGAN Decoder
    - Architecture as in paper
    - inpit: h x w x latent_space_dim
    - output: H x W x input_channels
'''
class Decoder(torch.nn.Module):
    def __init__(self, input_channels, latent_space_dim):
        super(Decoder, self).__init__()
        channel_dim = [512, 256, 256, 128, 128]

        # initial conv
        layers = [torch.nn.Conv2d(latent_space_dim, channel_dim[0], 3, 1, 1)]
        layers.append(ResidualBlock(channel_dim[0], channel_dim[0]))
        layers.append(NonLocalBlock(channel_dim[0]))
        layers.append(ResidualBlock(channel_dim[0]))

        # upsample blocks
        for i in range(len(channel_dim) - 1):
            in_channels = channel_dim[i]
            out_channels = channel_dim[i+1]

            layers.append(ResidualBlock(in_channels, out_channels))
            layers.append(ResidualBlock(out_channels, out_channels))
            layers.append(UpSample(out_channels))

        layers.append(GroupNorm(channel_dim[-1]))
        layers.append(Swish())
        layers.append(torch.nn.Conv2d(channel_dim[-1], input_channels, 3, 1, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)