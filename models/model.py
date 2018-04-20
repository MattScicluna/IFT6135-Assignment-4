"""
Implementation of a DCGAN-like architecture using deconvolution in the generator.
"""

import torch.nn as nn


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

    def __init__(self, hidden_dim, dropout=0.4, leaky=0.2):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=leaky),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=leaky),

            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=leaky),

            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=leaky),

            nn.ConvTranspose2d(in_channels=128, out_channels=3,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # forward
    def forward(self, x):
        return self.network(x)

    def weight_init(self, mean=0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):

    def __init__(self, dropout=0.4, leaky=0.2):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=leaky),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=leaky),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=leaky),

            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=leaky),

            nn.Conv2d(in_channels=1024, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    # forward
    def forward(self, x):
        return self.network(x)

    def weight_init(self, mean=0, std=0.02):
        for param in self._modules['network']:
            if isinstance(param, nn.Conv2d):
                nn.init.normal(param.weight, mean=mean, std=std)
                nn.init.constant(param.bias, 0.0)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()