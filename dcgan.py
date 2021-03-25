"""
2021/3/23 reproduce dcgan
mnist / cifar10 dataset
"""
import argparse
import os
import time

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from dataset import ImgDataset

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, help='project name', default='dcgan')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='./dataset/frs')
parser.add_argument('-input_size', type=int, help='input size of image', default=128)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.0002)
parser.add_argument('-epochs', type=int, help='training epochs', default=800)
parser.add_argument('-beta', type=float, help='beta', default=0.5)
parser.add_argument('-noise_size', type=int, help='noise size', default=100)
parser.add_argument('-nz', type=int, help='nz', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5,), (0.5,))  # convert (0,1) to (-1,1)
    ])

    # train_set = datasets.MNIST(
    #     args.dataset_path, train=True, download=True, transform=transform)
    # train_set = datasets.CIFAR10(
    #     args.dataset_path, train=True, download=True, transform=transform)

    class_label_dct = {'冬瓜排骨汤': 0, '土豆丝': 1, '椒盐虾': 2, '番茄炒蛋': 3, '糖醋里脊': 4, '红烧肉': 5, '莴笋肉片': 6, '辣子鸡': 7, '香菇青菜': 8,
                       '鱼香茄子': 9}
    train_set = ImgDataset(root=args.dataset_path, type_='train', transforms=transform, class_label_dct=class_label_dct,
                           num_per_class=200)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    return train_loader


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, args.input_size, args.input_size)
    return out


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator32(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator32, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. nc x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator32(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator32, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator128(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator128, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator128(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator128, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.name + now)
    os.makedirs(output_path)

    train_loader = load_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # generator = Generator32(nz=100, ngf=64, nc=3).to(device)
    # discriminator = Discriminator32(nc=3, ndf=64).to(device)

    # generator = Generator(nz=100, ngf=64, nc=3).to(device)
    # discriminator = Discriminator(nc=3, ndf=64).to(device)

    generator = Generator128(nz=100, ngf=64, nc=3).to(device)
    discriminator = Discriminator128(nc=3, ndf=64).to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta, 0.999))

    criterion = nn.BCELoss().to(device)

    for epoch in range(args.epochs):
        fake_img = None
        for idx, (inputs, _) in enumerate(train_loader):
            num_img = inputs.size(0)
            # convert tensor to variable
            real_img = Variable(inputs).to(device)

            # define label equal 1
            real_label = Variable(torch.ones(num_img)).cuda()  # batch_size * 1
            # define label equal 0
            fake_label = Variable(torch.zeros(num_img)).cuda()  # batch_size * 1

            # ====================train discriminator===================
            # calculate real loss
            real_out = discriminator(real_img).view(-1)
            d_loss_real = criterion(real_out, real_label)

            # calculate fake loss
            noise = Variable(torch.randn(num_img, args.nz, 1, 1)).cuda()  # batch_size * noise_size
            fake_img = generator(noise).detach()
            fake_out = discriminator(fake_img).view(-1)
            d_loss_fake = criterion(fake_out, fake_label)

            # back prop
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ====================train generator======================
            noise = Variable(torch.randn(num_img, args.nz, 1, 1)).cuda()
            fake_img = generator(noise)
            d_out = discriminator(fake_img).view(-1)

            g_loss = criterion(d_out, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (idx + 1) % 10 == 0:
                print('Epoch[{}/{}], D_loss:{:.6f}, G_loss:{:.6f}, D_real:{:.6f}, D_fake:{:.6f}'.format(
                    epoch, args.epochs, d_loss.data.item(), g_loss.data.item(),
                    real_out.data.mean(), fake_out.data.mean()))

        fake_imgs = to_img(fake_img.detach().cpu().data)
        save_image(fake_imgs, os.path.join(
            output_path, 'epoch-{}.png'.format(epoch)))

    # save model
    torch.save(generator.state_dict(), os.path.join(output_path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_path, 'discriminator.pth'))
