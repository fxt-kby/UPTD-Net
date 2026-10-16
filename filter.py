import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

#
# class FastGuidedFilter(nn.Module):
#     def __init__(self, r, eps=1e-5):
#         super(FastGuidedFilter, self).__init__()
#
#         self.r = r
#         self.eps = eps
#         self.boxfilter = BoxFilter(r)
#
#     def forward(self, lr_x, lr_y, hr_x):
#         n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
#         n_lry, c_lry, h_lry, w_lry = lr_y.size()
#         n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()
#
#         assert n_lrx == n_lry and n_lry == n_hrx
#         assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
#         assert h_lrx == h_lry and w_lrx == w_lry
#         assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1
#
#         # # N
#         N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))
#
#         # # mean_x
#         mean_x = self.boxfilter(lr_x) / N
#         # # mean_y
#         mean_y = self.boxfilter(lr_y) / N
#         # # cov_xy
#         cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
#         # # var_x
#         var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x
#
#         # # A
#         A = cov_xy / (var_x + self.eps)
#         # # b
#         b = mean_y - A * mean_x
#
#         # # mean_A; mean_b
#         mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#         mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#
#         return mean_A*hr_x+mean_b
#
#     # 作用：实现快速引导滤波器（Fast Guided Filter），用于图像增强、去噪、细节恢复等。
#     # self.r：滤波器半径； self.eps：防止除零的常数，通常设置为一个很小的值； self.boxfilter：调用 BoxFilter，实现卷积操作。
#     # forward 方法：
#     # 输入参数 lr_x, lr_y, hr_x 分别表示低分辨率输入图像、低分辨率引导图像和高分辨率输入图像。
#     # mean_x, mean_y, cov_xy, var_x：计算图像的均值、协方差和方差，用于后续的计算。
#     # A 和 b：利用均值和协方差计算滤波器的参数 A 和偏置项 b。
#     # F.interpolate：通过双线性插值将 A 和 b 的低分辨率结果上采样到高分辨率图像大小。
#     # 最后，滤波结果由 mean_A * hr_x + mean_b 得到。
#     # 通过引导图像和低分辨率图像计算出一个引导滤波器，然后用该滤波器对高分辨率图像进行增强或去噪操作。


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-7, small_eps=1e-10):
        super(FastGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.small_eps = small_eps  # 防止除零
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        # 尺寸检查
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # 验证输入
        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        # 归一化因子
        ones = Variable(torch.ones((1, 1, h_lrx, w_lrx), device=lr_x.device))
        N = self.boxfilter(ones)

        # 计算均值，添加小值防止数值问题
        mean_x = self.boxfilter(lr_x) / (N + self.small_eps)
        mean_y = self.boxfilter(lr_y) / (N + self.small_eps)

        # 计算协方差和方差，更稳健的实现
        mean_xy = self.boxfilter(lr_x * lr_y) / (N + self.small_eps)
        cov_xy = mean_xy - mean_x * mean_y

        mean_xx = self.boxfilter(lr_x * lr_x) / (N + self.small_eps)
        var_x = mean_xx - mean_x * mean_x

        # 关键修改1: 更稳定的自适应正则化参数 - 使用sigmoid来限制变化范围
        # 这有助于防止极端值导致的伪影
        var_x_normalized = torch.sigmoid(5 * var_x)  # 将方差映射到(0,1)范围
        adaptive_eps = self.eps * (1.0 - 0.7 * var_x_normalized) + 1e-8

        # 计算线性系数
        A = cov_xy / (var_x + adaptive_eps)
        b = mean_y - A * mean_x
        # 检测并修复可能的异常值
        A = torch.clamp(A, -8.0, 8.0)  # 之前是-20.0到20.0
        b = torch.clamp(b, -1.0, 2.0)  # 之前是-2.5到3.5

        # 上采样系数
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        # 应用系数并返回结果
        return torch.clamp(mean_A * hr_x + mean_b, 0.0, 1.0)