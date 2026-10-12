import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 定义 SirenLayer 类
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f  # 输入特征维度（即输入通道数）
        self.w0 = w0  # 正弦激活函数的缩放因子（默认值为30）
        self.linear = nn.Linear(in_f, out_f)  # 定义一个全连接层，将输入映射到输出维度
        self.is_first = is_first  # 标记该层是否为网络的第一层
        self.is_last = is_last  # 标记该层是否为网络的最后一层
        if not self.is_last:  # 如果不是最后一层，则初始化权重
            self.init_weights()

    # 如果是第一层，使用较小的值初始化权重；否则使用根据输入特征数量计算的初始化范围
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        # 第一层权重初始化：使用 1 / self.in_f，保证权重较小。
        # 其他层权重初始化：使用 np.sqrt(6 / self.in_f) / self.w0，这是一个基于输入维度的 He 或 Xavier 初始化方法，并加入了 w0 缩放因子来调节正弦激活函数的输出。
        with torch.no_grad():  # 禁用梯度计算
            self.linear.weight.uniform_(-b, b)  # 使用均匀分布初始化权重

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)  # 如果是最后一层，使用 Sigmoid 激活函数，否则使用 sin 激活函数

#
#   # img_siren = INF(patch_dim=opt.window**2, num_layers=4, hidden_dim=256, add_layer=2)
# class INF(nn.Module):
#     def __init__(self, patch_dim, num_layers, hidden_dim, add_layer, weight_decay=None):
#         super().__init__()
#         '''
#         `add_layer` should be in range of  [1, num_layers-2]
#         '''
#
#         patch_layers = [SirenLayer(patch_dim, hidden_dim, is_first=True)]  # 邻域像素网络的输入层，第一层
#         spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]  # 空间位置网络的输入层，第一层
#         output_layers = []  # 用于存储输出层
#
#         for _ in range(1, add_layer - 2):  # 逐层加入隐藏层add_layer=2没有新的隐藏层
#             patch_layers.append(SirenLayer(hidden_dim, hidden_dim))
#             spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))
#
#         patch_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))  # 添加一个新的层，并且输出维度减半>128
#         spatial_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))
#
#         for _ in range(add_layer, num_layers - 1):  # 加入输出层2 > 3,从2到3只循环一次
#             output_layers.append(SirenLayer(hidden_dim, hidden_dim))  # 添加一个输入输出维度为256的隐藏层
#         output_layers.append(SirenLayer(hidden_dim, 1, is_last=True))  # 添加最后一层，输入256输出单通道的结果
#
#         self.patch_net = nn.Sequential(*patch_layers)  # 将邻域像素网络的所有层组合成一个顺序容器
#         self.spatial_net = nn.Sequential(*spatial_layers)  # 将空间网络的所有层组合成一个顺序容器
#         self.output_net = nn.Sequential(*output_layers)  # 将输出网络的所有层组合成一个顺序容器
#
#         # 配置每个网络部分的参数和权重衰减
#         if not weight_decay:
#             weight_decay = [0.1, 0.0001, 0.001]
#
#         self.params = []  # 初始化一个空的列表 self.params，这个列表将存储网络中每个子网络的参数和对应的权重衰减值
#         self.params += [{'params': self.spatial_net.parameters(), 'weight_decay': weight_decay[0]}]  # 0.1
#         self.params += [{'params': self.patch_net.parameters(), 'weight_decay': weight_decay[1]}]  # 0.0001
#         self.params += [{'params': self.output_net.parameters(), 'weight_decay': weight_decay[2]}]  # 0.001
#
#     # 神经网络前向传播的过程，将输入的两个张量 patch 和 spatial 通过网络的不同部分进行处理，然后将它们拼接在一起，最终输出结果。
#     def forward(self, patch, spatial):
#         return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))  # 沿最后一个维度拼接在一起

# 修改后的INF网络，添加可学习的DOP权重参数

class INF(nn.Module):
    def __init__(self, patch_dim, num_layers, hidden_dim, add_layer, weight_decay=None,
                 initial_dop_weight=1.1):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''

        # ============= 添加可学习的DOP权重参数 =============
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

        # 配置每个网络部分的参数和权重衰减
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]

        self.params = []
        self.params += [{'params': self.spatial_net.parameters(), 'weight_decay': weight_decay[0]}]
        self.params += [{'params': self.patch_net.parameters(), 'weight_decay': weight_decay[1]}]
        self.params += [{'params': self.output_net.parameters(), 'weight_decay': weight_decay[2]}]

        # ============= 为DOP权重参数添加单独的学习率配置 =============
        self.params += [{'params': [self.dop_weight], 'weight_decay': 0.0, 'lr': 0.01}]  # 可以设置不同的学习率

    def forward(self, patch, spatial):
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))

    # ============= 添加获取DOP权重的方法 =============
    def get_dop_weight(self):
        """返回当前的DOP权重值，并确保在合理范围内"""
        return torch.clamp(self.dop_weight, 1.0, 2)  # 限制在0.5到2.0之间


# 设计D网络
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


# B 网络
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
    # 原始均为 5 1 2
