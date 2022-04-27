import torch
from codebook import VQEmbedding

class ResnetBlock(torch.nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = torch.nn.Sequential(torch.nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
                                              torch.nn.InstanceNorm2d(conv_dim))

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out

class Encoder(torch.nn.Module):
    def __init__(self, input_channels, latent_space_dim):
        super(Encoder, self).__init__()
        layers = []

        layers.append(torch.nn.Conv2d(input_channels, 32, 4, 2, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(32))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(64))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(128))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Conv2d(128, latent_space_dim, 4, 2, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(128))
        layers.append(torch.nn.ReLU())

        layers.append(ResnetBlock(latent_space_dim))
        layers.append(torch.nn.ReLU())
        layers.append(ResnetBlock(latent_space_dim))
        layers.append(torch.nn.ReLU())
        layers.append(ResnetBlock(latent_space_dim))
        layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class Decoder(torch.nn.Module):
    def __init__(self, input_channels, latent_space_dim):
        super(Decoder, self).__init__()

        # initial conv
        layers = []
        layers.append(ResnetBlock(latent_space_dim))
        layers.append(torch.nn.ReLU())
        layers.append(ResnetBlock(latent_space_dim))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(torch.nn.Conv2d(latent_space_dim, 128, 3, 1, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(32))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(torch.nn.Conv2d(128, 64, 3, 1, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(64))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(torch.nn.Conv2d(64, 32, 3, 1, 1, bias=False))
        layers.append(torch.nn.InstanceNorm2d(32))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(torch.nn.Conv2d(32, input_channels, 3, 1, 1, bias=False))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, input):
        return torch.nn.Tanh()(self.model(input))

class VQGAN(torch.nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args.input_channels, args.latent_space_dim)
        self.decoder = Decoder(args.input_channels, args.latent_space_dim)
        #self.codebook = CodeBook(args.num_vectors, args.latent_space_dim, args.beta)
        self.codebook = VQEmbedding(args.num_vectors, args.latent_space_dim)

        #self.enc_conv = torch.nn.Conv2d(args.latent_space_dim, args.latent_space_dim, 1)
        #self.code_conv = torch.nn.Conv2d(args.latent_space_dim, args.latent_space_dim, 1)

    def get_codebook(self):
        return self.codebook

    def encode(self, input):
        enc = self.encoder(input)
        z_q, idx = self.codebook(enc)
        return z_q, idx

    def decode(self, tokens):
        return self.decoder(tokens)

    def forward(self, input):
        encoded = self.encoder(input)
        #z_q, idx, loss = self.codebook(self.enc_conv(encoded))
        z_q, idx, loss = self.codebook.straight_through(encoded)
        #decoded = self.decoder(self.code_conv(z_q))
        decoded = self.decoder(z_q)

        return decoded, idx, loss
