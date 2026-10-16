import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        if not self.is_last:
            self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class INF(nn.Module):
    def __init__(self, patch_dim, num_layers, hidden_dim, add_layer, weight_decay=None,
                 initial_dop_weight=1.1):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''

        self.dop_weight = nn.Parameter(torch.tensor(initial_dop_weight, dtype=torch.float32))

        patch_layers = [SirenLayer(patch_dim, hidden_dim, is_first=True)]
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        output_layers = []

        for _ in range(1, add_layer - 2):
            patch_layers.append(SirenLayer(hidden_dim, hidden_dim))
            spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))

        patch_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))

        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SirenLayer(hidden_dim, hidden_dim))
        output_layers.append(SirenLayer(hidden_dim, 1, is_last=True))

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]

        self.params = []
        self.params += [{'params': self.spatial_net.parameters(), 'weight_decay': weight_decay[0]}]
        self.params += [{'params': self.patch_net.parameters(), 'weight_decay': weight_decay[1]}]
        self.params += [{'params': self.output_net.parameters(), 'weight_decay': weight_decay[2]}]

        self.params += [{'params': [self.dop_weight], 'weight_decay': 0.0, 'lr': 0.01}]

    def forward(self, patch, spatial):
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))

    def get_dop_weight(self):
        """返回当前的DOP权重值，并确保在合理范围内"""
        return torch.clamp(self.dop_weight, 1.0, 2)


class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        out_channel = 32
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 7, 1, 3),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, 1, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data2 = self.conv1(data)
        data2 = self.conv2(data2)
        data2 = self.conv3(data2)
        data2 = self.conv4(data2)
        D_out = self.final(data2)
        return D_out


class Net2(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        out_channel = 16
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, 1, 5, 1, 2),
            torch.nn.Sigmoid())

    def forward(self, data):
        data3 = self.conv1(data)
        data3 = self.conv2(data3)
        data3 = self.conv3(data3)
        data3 = self.conv4(data3)
        B_out = self.final(data3)
        return B_out

