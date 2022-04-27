import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
import PIL.Image as Image
from matplotlib import pyplot as plt
import torchvision

'''
    Perceptual Loss
'''
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # load pre-trained vgg16
        model = torchvision.models.vgg16(pretrained=True).features.eval()
        #self.blocks = [model[:4], model[:9], model[:16], model[:23]]
        self.blocks = [model[:4]]

        # normalization param
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, real, fake):
        # normalize images
        real = (real - self.mean) / self.std
        fake = (fake - self.mean) / self.std

        loss = 0

        for block in self.blocks:
            loss += torch.nn.MSELoss()(block(real), block(fake))

        return loss

'''
    Custom Data Loader
'''
class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.jpg', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, 0.

def get_data_loader(data_path, opts):
    """Creates training and test data loaders.
    """
    # transforms.Resize((256, 160), Image.BICUBIC),
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor()
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'deluxe':
        # todo: add your code here: below are some ideas for your reference
        load_size = int(1.1 * (182, 268))
        osize = [load_size, load_size]
        train_transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop((182, 268)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = CustomDataSet(os.path.join('data/', data_path), opts.ext, train_transform)
    dloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return dloader

'''
    For Displaying Images
'''
unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor1, tensor2):
    image1 = tensor1.cpu().clone()  # we clone the tensor to not do changes on it
    image1 = image1.squeeze(0)      # remove the fake batch dimension
    image1 = unloader(image1)

    image2 = tensor2.cpu().clone()
    image2 = image2.squeeze(0)
    image2 = unloader(image2)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image2)
    
    plt.pause(0.001)  # pause a bit so that plots are updated

def savefig(tensor1, tensor2, iter):
    image1 = tensor1.cpu().clone()  # we clone the tensor to not do changes on it
    image1 = image1.squeeze(0)      # remove the fake batch dimension
    image1 = unloader(image1)

    image2 = tensor2.cpu().clone()
    image2 = image2.squeeze(0)
    image2 = unloader(image2)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image2)

    f.savefig('reconst_' + str(iter) + '.jpg')

def savefig_1(tensor1, filename):
    image1 = tensor1.cpu().clone()  # we clone the tensor to not do changes on it
    image1 = image1.squeeze(0)      # remove the fake batch dimension
    image1 = unloader(image1)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image1)

    f.savefig(filename + '.jpg')

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