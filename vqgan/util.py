import torch

'''
    Swish layer
'''
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

'''
    Group Normalization
'''
class GroupNorm(torch.nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)

'''
    Residual Block from ResNet
     - with swish activation and group normalization
'''
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.out_trans = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, input):
        if self.in_channels != self.out_channels:
            return self.layers(input) + self.out_trans(input)
        
        return self.layers(input) + input

'''
    Downsample layer
'''
class DownSample(torch.nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, input):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(input, pad, mode="constant", value=0)
        return self.conv(x)

'''
    Upsample layer
'''
class UpSample(torch.nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0)
        return self.conv(x)


'''
    Non Local Block
'''
class NonLocalBlock(torch.nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A