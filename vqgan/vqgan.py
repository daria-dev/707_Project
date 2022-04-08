from encodings.utf_8 import encode
from turtle import forward
import torch
from encoder import Encoder
from decoder import Decoder
from codebook import CodeBook

class VQGAN(torch.nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args.input_channels, args.latent_space_dim)
        self.decoder = Decoder(args.input_channels, args.latent_space_dim)
        self.codebook - CodeBook(args.num_vectors, args.latent_space_dim)

        self.enc_conv = torch.nn.Conv2d(args.latent_space_dim, args.latent_space_dim, 1)
        self.code_conv = torch.nn.Conv2d(args.latent_space_dim, args.latent_space_dim, 1)

    def forward(self, input):
        encoded = self.encoder(input)
        z_q, idx, loss = self.codebook(self.enc_conv(encoded))
        decoded = self.decoder(self.code_conv(z_q))

        return decoded, idx, loss
