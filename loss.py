import torch
import torch.nn as nn
import torch.nn.functional as F


# 曝光度损失L_exp，基于图像局部区域的均值偏差损失。
# class L_exp(nn.Module):
#     def __init__(self, patch_size, mean_val):
#         super(L_exp, self).__init__()
#         self.pool = nn.AvgPool2d(patch_size)  # 使用平均池化层，池化窗口大小为 patch_size
#         self.mean_val = mean_val  # 存储期望的均值，这两个参数在主程序中已经设定，16和0.5
#
#     def forward(self, x):
#         mean = self.pool(x) ** 0.5
#         # 计算均值与期望均值之间的平方差，并对其求平均值的绝对值 l2范数不应该是平方和开根号吗
#         d = torch.abs(torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2)))
#         return d


class L_exp(nn.Module):
    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)  # 使用平均池化层，池化窗口大小为 patch_size

    def forward(self, x):
        mean = self.pool(x) ** 0.5
        # 计算均值与期望均值之间的平方差，并对其求平均值的绝对值 l2范数不应该是平方和开根号吗
        d = torch.abs(torch.mean(torch.pow(mean, 2)))
        return d


# 计算图像的总变差（Total Variation，简称 TV）损失
class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]  # 获取输入张量的 batch_size，h，w
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]  # 水平方向的像素数量
        count_w = x.size()[2] * (x.size()[3] - 1)  # 垂直方向的像素数量
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()  # 计算水平方向的总变差（像素间差异的平方和）
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()  # 计算垂直方向的总变差
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
# 返回总变差损失值（归一化后平均化）


def gradient_loss(pred, target):
    """计算预测图像和目标图像的梯度差异损失"""

    def compute_gradient(x):
        h_x = x.size()[2]
        w_x = x.size()[3]

        left = x[:, :, 1:h_x, 0:w_x - 1]
        right = x[:, :, 0:h_x - 1, 1:w_x]
        top = x[:, :, 0:h_x - 1, 0:w_x - 1]
        bottom = x[:, :, 1:h_x, 1:w_x]

        dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        return dx, dy

    dx_pred, dy_pred = compute_gradient(pred)
    dx_target, dy_target = compute_gradient(target)

    grad_loss = torch.mean(torch.abs(dx_pred - dx_target) + torch.abs(dy_pred - dy_target))
    return grad_loss


def direct_darkness_loss(B_out, target_darkness=0.01, weight=1.0):
    """
    直接约束背向散射光的平均灰度值保持较低

    参数:
        B_out: 模型输出的背向散射光 [batch, c, h, w]
        target_darkness: 目标平均灰度上限值，建议范围[0.1-0.2]
        weight: 损失权重系数

    返回:
        加权后的暗度损失
    """
    # 计算B_out的平均灰度值
    mean_intensity = torch.mean(B_out)

    # 设计损失：如果平均值高于目标值则惩罚，否则不惩罚
    # 使用ReLU确保只有当亮度超过目标时才产生梯度
    darkness_loss = F.relu(mean_intensity - target_darkness)

    # 对高于目标值的部分使用二次惩罚，使梯度随偏差增大而增强
    quadratic_loss = darkness_loss ** 2

    return weight * quadratic_loss


def fast_nuclear_norm_loss(x, weight=1.0):
    """
    使用Frobenius范数快速近似核范数的低秩损失函数

    参数:
        x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
        weight (float): 损失权重系数

    返回:
        torch.Tensor: 标量损失值
    """
    batch_size = x.size(0)
    loss = 0

    for i in range(batch_size):
        for j in range(x.size(1)):
            img = x[i, j, :, :]
            # 使用Frobenius范数作为近似
            frob_norm = torch.norm(img, p='fro')
            loss += frob_norm

    return weight * loss / batch_size