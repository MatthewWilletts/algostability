import torch
import torch.distributions as D
from torch.nn import functional as F
from torch import nn
from torch.distributions.utils import logits_to_probs, probs_to_logits
import torch.nn.utils.weight_norm as wn
import math
import numpy as np


class smoothReLU(nn.Module):
    """
    smooth ReLU activation function
    """

    def __init__(self, beta=1):
        super().__init__()
        self.beta = 1

    def forward(self, x):
        return x / (1 + torch.exp(-self.beta * x))


class LeafParam(nn.Module):
    """
    just ignores the input and outputs a parameter tensor
    """

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))


class PositionalEncoder(nn.Module):
    """
    Each dimension of the input gets expanded out with sins/coses
    to "carve" out the space. Useful in low-dimensional cases with
    tightly "curled up" data.
    """

    def __init__(self, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x


class MLP4(nn.Module):
    """ a simple 4-layer MLP4 """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class PosEncMLP(nn.Module):
    """
    Position Encoded MLP4, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """

    def __init__(self, nin, nout, nh, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP4(nin * len(freqs) * 2, nout, nh),
        )

    def forward(self, x):
        return self.net(x)


class MLPlayer(nn.Module):
    """
    implement basic module for MLP

    note that this module keeps the dimensions fixed! will implement a mapping from a
    vector of dimension input_size to another vector of dimension input_size
    """

    def __init__(self, input_size, output_size=None, activation_function=nn.functional.relu, use_bn=False):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.activation_function = activation_function
        self.linear_layer = nn.Linear(input_size, output_size)
        self.use_bn = use_bn
        self.bn_layer = nn.BatchNorm1d(input_size)

    def forward(self, x):
        if self.use_bn:
            x = self.bn_layer(x)
        linear_act = self.linear_layer(x)
        H_x = self.activation_function(linear_act)
        return H_x


class MLP(nn.Module):
    """
    define a MLP network - this is a more general class than MLP4 above, allows for user to specify
    the dimensions at each layer of the network
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size=None, activation_function=F.relu, use_bn=False):
        """
        Input:
         - input_size  : dimension of input data (e.g., 784 for MNIST)
         - hidden_size : list of hidden representations, one entry per layer
         - n_layers    : number of hidden layers
        """
        super().__init__()

        if output_size is None:
            output_size = 1  # because we approximating a log density, output should be scalar!

        self.use_bn = use_bn
        self.activation_function = activation_function
        self.linear1st = nn.Linear(input_size, hidden_size[0])  # map from data dim to dimension of hidden units
        self.Layers = nn.ModuleList([MLPlayer(hidden_size[i - 1], hidden_size[i],
                                              activation_function=self.activation_function, use_bn=self.use_bn) for i in
                                     range(1, n_layers)])
        self.linearLast = nn.Linear(hidden_size[-1],
                                    output_size)  # map from dimension of hidden units to dimension of output

    def forward(self, x):
        """
        forward pass through resnet
        """
        x = self.linear1st(x)
        for current_layer in self.Layers:
            x = current_layer(x)
        x = self.linearLast(x)
        return x


class CleanMLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, activation='lrelu', batch_norm=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.activation = activation
        self.batch_norm = batch_norm

        if activation == 'lrelu':
            act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            raise ValueError('wrong activation')

        # construct model
        if n_hidden == 0:
            modules = [nn.Linear(input_size, output_size)]
        else:
            modules = [nn.Linear(input_size, hidden_size), act] + batch_norm * [nn.BatchNorm1d(hidden_size)]

        for i in range(n_hidden - 1):
            modules += [nn.Linear(hidden_size, hidden_size), act] + batch_norm * [nn.BatchNorm1d(hidden_size)]

        modules += [nn.Linear(hidden_size, output_size)]

        self.net = nn.Sequential(*modules)

    def forward(self, x, y=None):
        return self.net(x)


class SimpleLinear(nn.Linear):
    """
    a wrapper around nn.Linear that defines custom fields
    """

    def __init__(self, nin, nout, bias=False):
        super().__init__(nin, nout, bias=bias)
        self.input_size = nin
        self.output_size = nout


class FullMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = self.input_size
        if config.model.final_layer:
            self.output_size = config.model.feature_size

        self.linear = nn.Sequential(
            nn.Linear(self.input_size, ngf * 8),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 8, ngf * 6),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Dropout(p=0.1),
            nn.Linear(ngf * 6, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, self.output_size)
        )

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.linear(output)
        return output


class FullMLPencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = self.input_size
        if config.model.final_layer:
            self.output_size = config.model.feature_size

        self.linear = nn.Sequential(
            nn.Linear(self.input_size, ngf * 8),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 8, ngf * 6),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Dropout(p=0.1),
            nn.Linear(ngf * 6, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
        )

        self.mean = nn.Linear(ngf * 4, self.output_size)
        self.log_var = nn.Linear(ngf * 4, self.output_size)

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.linear(output)
        return self.mean(output), self.log_var(output)


class FullMLPdecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = self.input_size
        if config.model.final_layer:
            self.output_size = config.model.feature_size

        self.linear = nn.Sequential(
            nn.Linear(self.output_size, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, ngf * 6),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 6, ngf * 8),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 8, self.input_size),
        )

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        return self.linear(output)


class ConvMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = im = config.data.image_size
        self.n_channels = nc = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = self.input_size
        if config.model.final_layer:
            self.output_size = config.model.feature_size

        # convolutional bit is [(conv, bn, relu, maxpool)*2, resize_conv)
        self.conv = nn.Sequential(
            # input is (nc, im, im)
            nn.Conv2d(nc, ngf // 2, 3, 1, 1),  # (ngf/2, im, im)
            nn.BatchNorm2d(ngf // 2),  # (ngf/2, im, im)
            nn.ELU(inplace=True),  # (ngf/2, im, im)
            nn.Conv2d(ngf // 2, ngf, 3, 1, 1),  # (ngf, im, im)
            nn.BatchNorm2d(ngf),  # (ngf, im, im)
            nn.ELU(inplace=True),  # (ngf, im, im)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (ngf, im/2, im/2)
            nn.Conv2d(ngf, ngf * 2, 3, 1, 1),  # (ngf*2, im/2, im/2)
            nn.BatchNorm2d(ngf * 2),  # (ngf*2, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*2, im/2, im/2)
            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1),  # (ngf*4, im/2, im/2)
            nn.BatchNorm2d(ngf * 4),  # (ngf*4, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*4, im/2, im/2)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (ngf*4, im/4, im/4)
            nn.Conv2d(ngf * 4, ngf * 4, im // 4, 1, 0)  # (ngf*4, 1, 1)
        )
        # linear bit is [drop, (lin, lrelu)*2, lin]
        self.linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(ngf * 4, ngf * 4),
            # nn.LeakyReLU(inplace=True, negative_slope=.1),
            # nn.Linear(ngf * 2, ngf * 2),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, self.output_size)
        )

    def forward(self, x):
        h = self.conv(x).squeeze()
        output = self.linear(h)
        return output


class ConvMLPencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = im = config.data.image_size
        self.n_channels = nc = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = config.model.feature_size

        # convolutional bit is [(conv, bn, relu, maxpool)*2, resize_conv)
        self.conv = nn.Sequential(
            # input is (nc, im, im)
            nn.Conv2d(nc, ngf // 2, 3, 1, 1),  # (ngf/2, im, im)
            nn.BatchNorm2d(ngf // 2),  # (ngf/2, im, im)
            nn.ELU(inplace=True),  # (ngf/2, im, im)
            nn.Conv2d(ngf // 2, ngf, 3, 1, 1),  # (ngf, im, im)
            nn.BatchNorm2d(ngf),  # (ngf, im, im)
            nn.ELU(inplace=True),  # (ngf, im, im)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (ngf, im/2, im/2)
            nn.Conv2d(ngf, ngf * 2, 3, 1, 1),  # (ngf*2, im/2, im/2)
            nn.BatchNorm2d(ngf * 2),  # (ngf*2, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*2, im/2, im/2)
            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1),  # (ngf*4, im/2, im/2)
            nn.BatchNorm2d(ngf * 4),  # (ngf*4, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*4, im/2, im/2)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (ngf*4, im/4, im/4)
            nn.Conv2d(ngf * 4, ngf * 4, im // 4, 1, 0)  # (ngf*4, 1, 1)
        )
        # linear bit is [drop, (lin, lrelu)*2, lin]
        self.linear = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(ngf * 4, ngf * 4),
            # nn.LeakyReLU(inplace=True, negative_slope=.1),
            # nn.Linear(ngf * 2, ngf * 2),
            nn.LeakyReLU(inplace=True, negative_slope=.1)
        )
        self.mean = nn.Linear(ngf * 4, self.output_size)
        self.log_var = nn.Linear(ngf * 4, self.output_size)

        # self.encoder = nn.Sequential(
        #     # input is mnist image: 1x28x28
        #     nn.Conv2d(config.data.channels, 32, 3, 1, 1),  # 32x16x16
        #     nn.BatchNorm2d(32),  # 32x16x16
        #     nn.ReLU(inplace=True),  # 32x16x16
        #     Interpolate(0.5),
        #     nn.Conv2d(32, 64, 3, 1, 1),  # 128x8x8
        #     nn.BatchNorm2d(64),  # 128x8x8
        #     nn.ReLU(inplace=True),  # 128x8x8
        #     Interpolate(0.5),
        #     nn.Conv2d(64, 128, 3, 1, 1),  # 128x4x4
        #     nn.BatchNorm2d(128),  # 128x4x4
        #     nn.ReLU(inplace=True),  # 128x4x4
        #     Interpolate(0.5),
        #     nn.Conv2d(128, 512, 3, 1, 1),  # 512x1x1
        #     nn.BatchNorm2d(512),  # 512x1x1
        #     nn.ReLU(inplace=True),  # 512x1x1
        #     Interpolate(0.5),
        #     nn.Conv2d(512, 200, 2, 1, 0),  # 200x1x1
        # )

        # self.encoder = nn.Sequential(
        #     # input is mnist image: 1x28x28
        #     nn.Conv2d(config.data.channels, 32, 4, 2, 1),  # 32x16x16
        #     nn.BatchNorm2d(32),  # 32x16x16
        #     nn.ReLU(inplace=True),  # 32x16x16
        #     nn.Conv2d(32, 64, 4, 2, 1),  # 128x8x8
        #     nn.BatchNorm2d(64),  # 128x8x8
        #     nn.ReLU(inplace=True),  # 128x8x8
        #     nn.Conv2d(64, 128, 4, 2, 1),  # 128x4x4
        #     nn.BatchNorm2d(128),  # 128x4x4
        #     nn.ReLU(inplace=True),  # 128x4x4
        #     nn.Conv2d(128, 512, 4, 1, 0),  # 512x1x1
        #     nn.BatchNorm2d(512),  # 512x1x1
        #     nn.ReLU(inplace=True),  # 512x1x1
        #     nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        # )
        # self.fc1 = nn.Linear(200, self.output_size)
        # self.fc2 = nn.Linear(200, self.output_size)

    def forward(self, x):
        h = self.conv(x).squeeze()
        output = self.linear(h)
        return self.mean(output), self.log_var(output)
        # h = self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size)).squeeze()
        # return self.fc1(F.relu(h)), self.fc2(F.relu(h))


class DeconvMLPdecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = im = config.data.image_size
        self.n_channels = nc = config.data.channels
        self.ngf = ngf = config.model.ngf

        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.output_size = config.model.feature_size

        # self.fc3 = nn.Linear(self.output_size, 200)

        # self.decoder = nn.Sequential(
        #     # input: latent_dim x 1 x 1
        #     nn.Conv2d(200, 512, 1, 1, 0),  # 512x1x1
        #     nn.BatchNorm2d(512),  # 512x1x1
        #     nn.ReLU(inplace=True),  # 512x1x1
        #     Interpolate(2),
        #     nn.Conv2d(512, 128, 3, 1, 1),  # 128x4x4
        #     nn.BatchNorm2d(128),  # 128x7x7
        #     nn.ReLU(inplace=True),  # 128x7x7
        #     Interpolate(2),
        #     nn.Conv2d(128, 64, 3, 1, 1),  # 32x8x8
        #     nn.BatchNorm2d(64),  # 32x14x14
        #     nn.ReLU(inplace=True),  # 32x14x14
        #     Interpolate(2),
        #     nn.Conv2d(64, 64, 3, 1, 1),  # 32x8x8
        #     nn.BatchNorm2d(64),  # 32x14x14
        #     nn.ReLU(inplace=True),  # 32x14x14
        #     Interpolate(2),
        #     nn.Conv2d(64, 32, 3, 1, 1),  # 32x16x16
        #     nn.BatchNorm2d(32),  # 32x14x14
        #     nn.ReLU(inplace=True),  # 32x14x14
        #     Interpolate(2),
        #     nn.Conv2d(32, config.data.channels, 3, 1, 1)  # 1x28x28
        # )

        # self.decoder = nn.Sequential(
        #     # input: latent_dim x 1 x 1
        #     nn.ConvTranspose2d(200, 512, 1, 1, 0),  # 512x1x1
        #     nn.BatchNorm2d(512),  # 512x1x1
        #     nn.ReLU(inplace=True),  # 512x1x1
        #     nn.ConvTranspose2d(512, 128, 4, 1, 0),  # 128x4x4
        #     nn.BatchNorm2d(128),  # 128x7x7
        #     nn.ReLU(inplace=True),  # 128x7x7
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x8x8
        #     nn.BatchNorm2d(64),  # 32x14x14
        #     nn.ReLU(inplace=True),  # 32x14x14
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x16x16
        #     nn.BatchNorm2d(32),  # 32x14x14
        #     nn.ReLU(inplace=True),  # 32x14x14
        #     nn.ConvTranspose2d(32, config.data.channels, 4, 2, 1)  # 1x28x28
        # )
        # deconvolutional bit is [(deconv, bn, relu, upscale)*2, resize_conv)
        self.conv = nn.Sequential(
            # input is (ngf*4, 1, 1)
            nn.ConvTranspose2d(ngf * 4, ngf * 4, im // 4, 1, 0),  # (ngf*4, im/4, im/4)
            nn.BatchNorm2d(ngf * 4),  # (ngf*4, im/4, im/4)
            nn.ELU(inplace=True),  # (ngf*4, im/4, im/4)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # (ngf*2, im/2, im/2)
            nn.BatchNorm2d(ngf * 2),  # (ngf*2, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*2, im/2, im/2)
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 1, 1),  # (ngf, im/2, im/2)
            nn.BatchNorm2d(ngf),  # (ngf*2, im/2, im/2)
            nn.ELU(inplace=True),  # (ngf*2, im/2, im/2)
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),  # (ngf/2, im, im)
            nn.BatchNorm2d(ngf // 2),  # (ngf/2, im, im)
            nn.ELU(inplace=True),  # (ngf/2, im, im)
            nn.Conv2d(ngf // 2, nc, 3, 1, 1))

        # reverse linear bit is [drop, (lin, lrelu)*2, lin]
        self.linear = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(self.output_size, ngf * 4),
            nn.LeakyReLU(inplace=True, negative_slope=.1),
            nn.Linear(ngf * 4, ngf * 4),
            # nn.LeakyReLU(inplace=True, negative_slope=.1),
            # nn.Linear(ngf * 2, ngf * 2),
        )

    def forward(self, z):
        h = self.linear(z)
        return self.conv(h.view(h.size(0), h.size(1), 1, 1))
        # z = self.fc3(z)
        # return self.decoder(z.view(z.size(0), z.size(1), 1, 1))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, resample=None, activation=F.relu,
                 dropout_prob=0., kernel_size=3, resample_stride=2, first=False,
                 identity_rescale=False):
        super().__init__()
        self.in_channels = in_channels
        self.resample = resample
        self.activation = activation
        self.identity_rescale = identity_rescale

        self.residual_layer_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=1
        )
        self.batch_norm_1 = nn.BatchNorm2d(in_channels)

        if resample is None:
            self.shortcut_layer = nn.Identity()
            self.residual_2_layer = wn(nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=1
            ))
            self.batch_norm_2 = nn.BatchNorm2d(in_channels)
        elif resample == 'down':
            if self.identity_rescale:
                self.shortcut_layer = Interpolate(scale_factor=1.0/resample_stride)
            else:
                self.shortcut_layer = wn(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=kernel_size,
                    stride=resample_stride,
                    padding=1
                ))
            self.residual_2_layer = wn(nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=kernel_size,
                stride=resample_stride,
                padding=1
            ))
            self.batch_norm_2 = nn.BatchNorm2d(in_channels * 2)
        elif resample == 'up':
            if self.identity_rescale:
                self.shortcut_layer = Interpolate(scale_factor=resample_stride)
            else:
                self.shortcut_layer = wn(nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=kernel_size,
                    stride=resample_stride,
                    padding=1,
                    output_padding=0 if first else 1
                ))
            self.residual_2_layer = wn(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                stride=resample_stride,
                padding=1,
                output_padding=0 if first else 1
            ))
            self.batch_norm_2 = nn.BatchNorm2d(in_channels // 2)

        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inputs):

        shortcut = self.shortcut_layer(inputs)
        residual_1 = self.activation(inputs)
        residual_1 = self.residual_layer_1(residual_1)
        # residual_1 = self.batch_norm_1(residual_1)
        if self.dropout is not None:
            residual_1 = self.dropout(residual_1)
        residual_2 = self.activation(residual_1)
        residual_2 = self.residual_2_layer(residual_2)
        # residual_2 = self.batch_norm_2(residual_2)

        return shortcut + residual_2


class ResnetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = im = config.data.image_size
        self.n_channels = in_channels = config.data.channels
        self.channels_multiplier = channels_multiplier = config.model.ngf
        self.activation = F.relu

        self.n_stacks = n_stacks = 5
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.z_dim = config.model.feature_size

        if config.data.image_size == 28:
            self.height_dims = [28, 14, 7, 4, 2, 1]
        elif config.data.image_size == 32:
            self.height_dims = [32, 16, 8, 4, 2, 1]

        non_dim_change_layer = False
        dropout_prob = 0.

        if config.data.image_size == 28:
            self.height_dims = [28, 14, 7, 4, 2, 1]
        elif config.data.image_size == 32:
            self.height_dims = [32, 16, 8, 4, 2, 1]

        self.initial_layer = nn.Conv2d(in_channels=in_channels, out_channels=channels_multiplier, kernel_size=1)
        self.residual_blocks = nn.ModuleList([])

        for i in range(n_stacks):
            if non_dim_change_layer:
                # previously 3 stacks
                self.residual_blocks.append(ResidualBlock(in_channels=channels_multiplier * (2 ** i), dropout_prob=dropout_prob))
            self.residual_blocks.append(ResidualBlock(in_channels=channels_multiplier * (2 ** i), resample='down', dropout_prob=dropout_prob))

        self.mean = nn.Linear((self.height_dims[self.n_stacks] * self.height_dims[self.n_stacks]) * (self.channels_multiplier * (2 ** self.n_stacks)), self.z_dim)
        self.log_var = nn.Linear((self.height_dims[self.n_stacks] * self.height_dims[self.n_stacks]) * (self.channels_multiplier * (2 ** self.n_stacks)), self.z_dim)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)

        outputs = temps.reshape(-1, (self.height_dims[self.n_stacks] * self.height_dims[self.n_stacks]) * (self.channels_multiplier * (2 ** self.n_stacks)))  # (4 * 4) is height*width, (self.channels_multiplier * 8) are the output channels of the last layer
        return self.mean(outputs), self.log_var(outputs)


class ResnetDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_classes = config.model.num_classes
        self.image_size = im = config.data.image_size
        self.n_channels = in_channels = config.data.channels
        self.channels_multiplier = channels_multiplier = config.model.ngf
        self.activation = F.relu

        self.n_stacks = n_stacks = 5
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.z_dim = z_dim = config.model.feature_size

        if config.data.image_size == 28:
            self.height_dims = [28, 14, 7, 4, 2, 1]
        elif config.data.image_size == 32:
            self.height_dims = [32, 16, 8, 4, 2, 1]

        non_dim_change_layer = False
        dropout_prob = 0.

        self.initial_layer = nn.Linear(
            in_features=z_dim,
            out_features=((self.height_dims[n_stacks] * self.height_dims[n_stacks]) * channels_multiplier * (2 ** self.n_stacks))
        )
        self.residual_blocks = nn.ModuleList([])

        for i in range(n_stacks):
            if non_dim_change_layer:
                # previously 3 stacks, all channel dims are symmetric to encoder
                self.residual_blocks.append(ResidualBlock(in_channels=channels_multiplier * (2 ** (n_stacks - i)), dropout_prob=dropout_prob))
            self.residual_blocks.append(ResidualBlock(in_channels=channels_multiplier * (2 ** (n_stacks - i)), resample='up',
                                                      first=True if i == 2 and config.data.image_size == 28 else False, dropout_prob=dropout_prob))

        self.final_layer = nn.Conv2d(channels_multiplier * (2 ** (n_stacks - i - 1)), in_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        temps = self.initial_layer(inputs).reshape(-1, self.channels_multiplier * (2 ** self.n_stacks), self.height_dims[self.n_stacks], self.height_dims[self.n_stacks])
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        outputs = self.activation(temps)
        return self.final_layer(outputs)
