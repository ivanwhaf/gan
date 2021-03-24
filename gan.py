"""
2021/3/23 reproduce gan
mnist / cifar10 dataset
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, help='project name', default='gan')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='./dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.0003)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
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


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            # nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            # nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.name + now)
    os.makedirs(output_path)

    train_loader = load_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # generator = Generator(args.noise_size, 256, 784).to(device)
    # discriminator = Discriminator(784, 256, 1).to(device)  # mnist 28*28

    generator = Generator(args.noise_size, 1024, 3072).to(device)
    discriminator = Discriminator(3072, 1024, 1).to(device)  # cifar10 32*32*3

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

    criterion = nn.BCELoss().to(device)

    g_loss_lst, d_loss_lst = [], []

    for epoch in range(args.epochs):
        fake_img = None
        for idx, (inputs, _) in enumerate(train_loader):
            num_img = inputs.size(0)
            # flatten imgs
            inputs = inputs.view(num_img, -1)  # batch_size * wh

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
            fake_img = generator(noise).detach()  # better add detach
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

            g_loss_lst.append(g_loss.item())
            d_loss_lst.append(d_loss.item())

        fake_imgs = to_img(fake_img.detach().cpu().data)
        save_image(fake_imgs, os.path.join(
            output_path, 'epoch-{}.png'.format(epoch)))

    # plot loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss_lst, label="G")
    plt.plot(d_loss_lst, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # save model
    torch.save(generator.state_dict(), os.path.join(output_path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_path, 'discriminator.pth'))
