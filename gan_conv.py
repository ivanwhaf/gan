"""
2021/3/23 reproduce conv gan
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
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, help='project name', default='gan_conv')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='./dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.0003)
parser.add_argument('-epochs', type=int, help='training epochs', default=1000)
parser.add_argument('-noise_size', type=int, help='noise size', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert (0,255) to (0,1)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5,), (0.5,))  # convert (0,1) to (-1,1)
    ])

    # train_set = datasets.MNIST(
    #     args.dataset_path, train=True, download=True, transform=transform)
    train_set = datasets.CIFAR10(
        args.dataset_path, train=True, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    return train_loader


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 32, 32)
    # out = out.view(-1, 1, 28, 28)
    return out


class GeneratorConv(nn.Module):
    def __init__(self, input_size=100, feature_map_size=56, out_channel=1):
        super(GeneratorConv, self).__init__()
        self.feature_map_size = feature_map_size
        self.out_channel = out_channel

        self.fc = nn.Linear(input_size, self.feature_map_size ** 2)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, self.out_channel, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, self.feature_map_size, self.feature_map_size)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


class DiscriminatorConv(nn.Module):
    def __init__(self, input_channel=1, img_size=28):
        super(DiscriminatorConv, self).__init__()
        self.input_channel = input_channel
        self.img_size = img_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.img_size // 4) * (self.img_size // 4), 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: batch, width, height, channel=1
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.name + now)
    os.makedirs(output_path)

    train_loader = load_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = GeneratorConv(input_size=args.noise_size, feature_map_size=32 * 2, out_channel=3).to(device)
    discriminator = DiscriminatorConv(input_channel=3, img_size=32).to(device)  # mnist 28*28

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

    criterion = nn.BCELoss().to(device)

    for epoch in range(args.epochs):
        fake_img = None
        for idx, (inputs, _) in enumerate(train_loader):
            num_img = inputs.size(0)

            # convert tensor to variable
            real_img = Variable(inputs).to(device)
            # define label equal 1
            real_label = Variable(torch.ones(num_img, 1)).cuda()  # batch_size * 1
            # define label equal 0
            fake_label = Variable(torch.zeros(num_img, 1)).cuda()  # batch_size * 1

            # ====================train discriminator===================
            # calculate real loss
            real_out = discriminator(real_img)
            d_loss_real = criterion(real_out, real_label)

            # calculate fake loss
            noise = Variable(torch.randn(num_img, args.noise_size)).cuda()  # batch_size * noise_size
            fake_img = generator(noise).detach()
            fake_out = discriminator(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)

            # back prop
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ====================train generator======================
            noise = Variable(torch.randn(num_img, args.noise_size)).cuda()
            fake_img = generator(noise)
            d_out = discriminator(fake_img)

            g_loss = criterion(d_out, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (idx + 1) % 200 == 0:
                print('Epoch[{}/{}], D_loss:{:.6f}, G_loss:{:.6f}, D_real:{:.6f}, D_fake:{:.6f}'.format(
                    epoch, args.epochs, d_loss.data.item(), g_loss.data.item(),
                    real_out.data.mean(), fake_out.data.mean()))

        fake_imgs = to_img(fake_img.detach().cpu().data)
        save_image(fake_imgs, os.path.join(
            output_path, 'epoch-{}.png'.format(epoch)))

    # save model
    torch.save(generator.state_dict(), os.path.join(output_path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_path, 'discriminator.pth'))
