import torch

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1)
        norm1 = torch.nn.BatchNorm2d(32)

        conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1)
        norm2 = torch.nn.BatchNorm2d(64)

        conv3 = torch.nn.Conv2d(64, 128, 4, 2, 1)
        norm3 = torch.nn.BatchNorm2d(128)

        conv4 = torch.nn.Conv2d(128, 1, 4, 2, 1)

        self.model = torch.nn.Sequential(conv1, norm1, torch.nn.ReLU(),
                                    conv2, norm2, torch.nn.ReLU(),
                                    conv3, norm3, torch.nn.ReLU(),
                                    conv4
                                   )

    def forward(self, input):
        out = self.model(input)
        prob = torch.nn.Sigmoid()(out)

        return prob