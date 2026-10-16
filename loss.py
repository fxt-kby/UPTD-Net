import torch
import torch.nn as nn
import torch.nn.functional as F


class L_exp(nn.Module):
    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x):
        mean = self.pool(x) ** 0.5
        d = torch.abs(torch.mean(torch.pow(mean, 2)))
        return d


class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


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
    mean_intensity = torch.mean(B_out)
    darkness_loss = F.relu(mean_intensity - target_darkness)
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