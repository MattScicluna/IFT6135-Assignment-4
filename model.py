import torch
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
    '''
    Following carefully the wisdom of: from https://arxiv.org/pdf/1511.06434.pdf
    Architecture guidelines for stable Deep Convolutional GANs:
        • Replace any pooling layers with strided convolutions (discriminator) and
          fractional-strided convolutions (generator).
        • Use batchnorm in both the generator and the discriminator.
        • Remove fully connected hidden layers for deeper architectures.
        • Use ReLU activation in generator for all layers except for the output, which uses Tanh.
        • Use LeakyReLU activation in the discriminator for all layers.
    '''

    def __init__(self, hidden_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                           kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=3,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # forward
    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=1024, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    # forward
    def forward(self, x):
        return self.network(x)
