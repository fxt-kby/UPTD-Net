import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from filter import FastGuidedFilter


# 黑白获取图像并进行预处理   读取指定路径的图像文件，转换为 PyTorch 张量，归一化到 [0, 1] ，修改图像维度(1,3,H,W)返回。
# def get_image(path):
#     """
#     Reads and returns RGB image, (1,3,H,W).
#     """
#     image = (np.asarray(Image.open(path)) / 255.0)
#     image = torch.from_numpy(image).float()
#     # image = image[100:1024, 100:1024]     #  [:, 200:1224] [100:, 100:1024]  [:, :1024]
#     image = image.unsqueeze(2)
#     image = torch.movedim(image, -1, 0).unsqueeze(0).cuda()  # 调整维度顺序 (H,W,C) -> (1,C,H,W)，并将其移到 GPU 上
#     return image
#临时
def get_image(path, device='cuda'):
    """
    读取图像并返回 4D Tensor (1, 1, H, W)，强制灰度。
    """
    img = Image.open(path).convert('L')           # 强制灰度
    arr = np.array(img).astype('float32') / 255.0  # (H, W)
    arr = arr[None, ...]                           # (1, H, W) -> channel-first
    tensor = torch.from_numpy(arr).unsqueeze(0)    # (1, 1, H, W)
    if device == 'cuda' and torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# 从 HSV 图像中提取亮度（V）通道
def get_v_component(img_hsv):
    """
    Assumes (1,3,H,W) HSV image.
    """
    return img_hsv[:, -1].unsqueeze(0)  # 提取HSV图像中最后一个通道（-1）亮度V通道，并在最前面增加一个维度


# 替换 HSV 图像中的 V 通道（亮度）
def replace_v_component(img_hsv, v_new):
    """
    Replaces the V component of a HSV image (1,3,H,W).
    """
    img_hsv[:, -1] = v_new
    return img_hsv


# 将图像大小调整为新的分辨率 (H, W)
def interpolate_image(img, H, W):
    """
    Reshapes the image based on new resolution.
    """
    return F.interpolate(img, size=(H, W))  # 使用 PyTorch 的插值功能将图像重采样到新的尺寸 (H, W)


# 生成一个大小为 (H, W) 的坐标网格，用于神经网络模型（例如 SIREN 模型）的输入。
def get_coords(H, W):
    """
    Creates a coordinates grid for INF.
    """
    # np.meshgrid(np.linspace(0, 1, H), ...(0, 1, W)) 创建一个从 0 到 1 的均匀坐标网格，大小为 (H, W)。使用 np.dstack 将两个一维坐标轴组合成一个 2D 网格坐标。
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    # 转换为浮点型 PyTorch 张量并移动到 GPU
    coords = torch.from_numpy(coords).float().cuda()
    return coords


# 创建一个包含图像补丁信息的张量。KERNEL_SIZE对应上下文窗口尺寸
def get_patches(img, KERNEL_SIZE):
    """
    Creates a tensor where the channel contains patch information.
    """
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()  # 创建一个 (KERNEL_SIZE x KERNEL_SIZE)卷积核

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1  # 将卷积核的最后一个位置设为 1，表示生成的补丁区域

    pad = nn.ReflectionPad2d(KERNEL_SIZE // 2)  # 为图像添加对称填充，使得卷积操作可以提取图像的边缘区域。
    im_padded = pad(img)  # 将图像进行填充
    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)  # 对图像进行卷积，移除批次维度
    # (1, out_channels, H_out, W_out)>(out_channels, H_out, W_out)

    return torch.movedim(extracted, 0, -1)  # 调整补丁张量的维度，>( H_out, W_out，out_channels)


# 使用引导滤波器将低分辨率图像 y_lr 上采样到高分辨率图像 x_hr。
def filter_up(x_lr, y_lr, x_hr, r=2):
    """
    Applies the guided filter to upscale the predicted image.
    """
    # 添加输入数据验证和预处理
    x_lr = torch.nan_to_num(x_lr, nan=0.0, posinf=0.99, neginf=0.01)
    y_lr = torch.nan_to_num(y_lr, nan=0.0, posinf=0.99, neginf=0.01)
    x_hr = torch.nan_to_num(x_hr, nan=0.0, posinf=0.99, neginf=0.01)
    guided_filter = FastGuidedFilter(r=r)  # 创建一个快速引导滤波器对象，r是滤波器的半径。filter.py文件
    y_hr = guided_filter(x_lr, y_lr, x_hr)  # 使用引导滤波器对低分辨率图像 y_lr 和参考图像 x_lr 进行滤波，生成上采样后的高分辨率图像 y_hr。
    y_hr = torch.clip(y_hr, 0, 1)  # 对输出图像进行裁剪，确保其像素值在 [0, 1] 范围内
    return y_hr  # 返回上采样后的高分辨率图像y_hr


def np_to_torch(img_np):
    return torch.from_numpy(img_np).unsqueeze(0)


def prepare_image(file_path):
    image = Image.open(file_path).convert('L')  # 转换为灰度图像
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # 增加通道维度
    return image


def gaussian_low_pass_filter(image_tensor, sigma=3):
    """
    对图像张量进行高斯低通滤波
    :param image_tensor: 输入图像张量 [batch_size, channels, height, width]
    :param sigma: 高斯核的标准差
    :return: 滤波后的图像张量 默认sigma=3

    """
    # 检查输入维度
    if len(image_tensor.shape) != 4:
        raise ValueError("Expected 4D tensor [batch_size, channels, height, width]")

    device = image_tensor.device  # 移动到指定设备
    B, C, H, W = image_tensor.shape  # 获取图像大小
    x = torch.linspace(-W // 2, W // 2, W, device=device)
    y = torch.linspace(-H // 2, H // 2, H, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    D = torch.sqrt(X ** 2 + Y ** 2)
    H = torch.exp(-(D ** 2) / (2 * sigma ** 2))
    H = H.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # FFT处理
    with torch.amp.autocast('cuda'):
        f = torch.fft.fft2(image_tensor)
        fshift = torch.fft.fftshift(f)
        fshift_filtered = fshift * H
        f_ishift = torch.fft.ifftshift(fshift_filtered)
        img_filtered = torch.abs(torch.fft.ifft2(f_ishift))

    return img_filtered


def mean_filter_tensor(tensor, kernel_size):
    """
    对PyTorch张量应用均值滤波(Mean/Average Filter)

    参数:
        tensor: 输入张量，形状为 [B, C, H, W]
        kernel_size: 滤波器窗口大小，必须是奇数

    返回:
        应用均值滤波后的张量，形状与输入相同
    """
    # 检查输入
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量")

    # 记住原始设备
    device = tensor.device

    # 将张量转换为numpy数组
    tensor_np = tensor.detach().cpu().numpy()

    # 创建输出数组
    filtered = np.zeros_like(tensor_np)

    # 对每个批次和每个通道应用均值滤波
    for b in range(tensor_np.shape[0]):
        for c in range(tensor_np.shape[1]):
            # 应用均值滤波 - cv2.blur是均值滤波
            filtered[b, c] = cv2.blur(tensor_np[b, c].astype(np.float32), (kernel_size, kernel_size))

    # 转回PyTorch张量并移回原始设备
    filtered_tensor = torch.from_numpy(filtered).to(device)

    return filtered_tensor


#########################################################################################################################
# 细节和亮度增强分支相关函数
# =========== 图像增强相关函数 ===========

def calculate_eme(img, block_size=8):
    """
    计算图像的EME (Enhancement Measure Evaluation) 值

    参数:
        img: 输入图像 (可以是tensor或numpy数组)
        block_size: 分块大小

    返回:
        EME值
    """
    # 如果输入是tensor，转换为numpy
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.detach().cpu()
        if img.dim() == 4:  # [B,C,H,W]
            img = img[0, 0].numpy()  # 取第一张图的第一个通道
        elif img.dim() == 3:  # [C,H,W]
            img = img[0].numpy()
        else:
            img = img.numpy()

    # 确保是灰度图
    if len(img.shape) > 2:
        if img.shape[2] == 3 or img.shape[0] == 3:  # HWC or CHW
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 归一化到[0,255]
    if img.max() <= 1.0:
        img = img * 255.0

    # 防止除以零，将零值替换为一个很小的值
    img = np.maximum(img.astype(np.float32), 0.001)

    h, w = img.shape
    k1 = h // block_size
    k2 = w // block_size

    sum_eme = 0

    # 遍历所有块
    for i in range(k1):
        for j in range(k2):
            block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            block_max = np.max(block)
            block_min = np.min(block)

            # 避免除以零或取对数时出错
            if 0 < block_min < block_max:
                sum_eme += 20.0 * np.log10(block_max / block_min)

    # 计算平均EME值
    eme = sum_eme / (k1 * k2) if k1 * k2 > 0 else 0
    return eme


def create_brightness_weight_map(img_B, threshold_factor=1, smoothness=0):    # threshold_factor=1.0
    """
    创建亮度权重图，只在高于平均亮度的区域有权重

    参数:
        img_B: 输入图像 (numpy数组或tensor)
        threshold_factor: 亮度阈值因子，默认为1.0表示使用平均亮度作为阈值
        smoothness: 权重图平滑因子

    返回:
        亮度权重图，高于平均亮度的区域有权重，低于的区域权重为0
    """
    # 将tensor转换为numpy处理
    is_tensor = isinstance(img_B, torch.Tensor)
    device = None

    if is_tensor:
        device = img_B.device
        # 处理不同形状的tensor
        if img_B.dim() == 4:  # [B,C,H,W]
            img_B_np = img_B[0, 0].detach().cpu().numpy()
        elif img_B.dim() == 3:  # [C,H,W]
            img_B_np = img_B[0].detach().cpu().numpy()
        else:  # [H,W]
            img_B_np = img_B.detach().cpu().numpy()
    else:
        img_B_np = img_B

    # 确保输入是灰度图
    if len(img_B_np.shape) > 2:
        img_B_np = cv2.cvtColor(img_B_np, cv2.COLOR_RGB2GRAY)

    # 归一化到[0,255]
    if img_B_np.max() <= 1.0:
        img_B_np = img_B_np * 255.0

    # 计算平均亮度作为阈值
    mean_brightness = np.mean(img_B_np)
    threshold = mean_brightness * threshold_factor

    # 创建亮度权重图
    brightness_map = np.zeros_like(img_B_np, dtype=np.float32)

    # 高于阈值的区域设置权重，低于的设为0
    mask1 = img_B_np > threshold
    brightness_map[mask1] = (img_B_np[mask1] - threshold)/ (255.0 - threshold)
    mask2 = img_B_np < threshold                                                                        # 在跑20000时使用
    brightness_map[mask2] = ((img_B_np[mask2]-threshold) / (255.0 - threshold)) * 0.4

    brightness_map = cv2.normalize(brightness_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 平滑权重图
    if smoothness > 0:
        brightness_map = cv2.GaussianBlur(brightness_map, (0, 0), smoothness)

    # 归一化到[0,1]范围
    brightness_map = cv2.normalize(brightness_map, None, 0, 1, cv2.NORM_MINMAX)

    # 如果输入是tensor，返回tensor
    if is_tensor:
        brightness_map = torch.from_numpy(brightness_map).float()
        if device:
            brightness_map = brightness_map.to(device)
        # 根据原始tensor形状调整输出
        if img_B.dim() == 4:
            brightness_map = brightness_map.unsqueeze(0).unsqueeze(0)
        elif img_B.dim() == 3:
            brightness_map = brightness_map.unsqueeze(0)

    return brightness_map


def create_advanced_weight_map(img_B):
    """
    创建一个结合梯度特征和局部熵特征的自适应权重图，表示图像的细节丰富程度

    参数:
        img_B: 输入图像，应当是细节丰富的参考图像 (numpy数组或tensor)

    返回:
        权重图，值范围[0,1]，细节丰富区域权重大
    """
    # 将tensor转换为numpy处理
    is_tensor = isinstance(img_B, torch.Tensor)
    device = None

    if is_tensor:
        device = img_B.device
        # 处理不同形状的tensor
        if img_B.dim() == 4:  # [B,C,H,W]
            img_B_np = img_B[0, 0].detach().cpu().numpy()
        elif img_B.dim() == 3:  # [C,H,W]
            img_B_np = img_B[0].detach().cpu().numpy()
        else:  # [H,W]
            img_B_np = img_B.detach().cpu().numpy()
    else:
        img_B_np = img_B

    # 确保输入是灰度图
    if len(img_B_np.shape) > 2:
        img_B_np = cv2.cvtColor(img_B_np, cv2.COLOR_RGB2GRAY)

    # 归一化到[0,255]
    if img_B_np.max() <= 1.0:
        img_B_np = img_B_np * 255.0

    # 转换为浮点数进行计算
    img_B_float = img_B_np.astype(np.float32)

    # 1. 梯度特征计算 - 使用Sobel算子检测边缘和纹理
    grad_x = cv2.Sobel(img_B_float, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img_B_float, cv2.CV_32F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_feature = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # 2. 局部熵特征（衡量区域信息量/复杂度）
    entropy_kernel_size = 5  # 熵计算的窗口大小
    entropy = np.zeros_like(img_B_float)
    h, w = img_B_np.shape

    # 为提高效率，每隔2个像素计算一次，然后扩展到整个区域
    for i in range(0, h - entropy_kernel_size, 2):
        for j in range(0, w - entropy_kernel_size, 2):
            patch = img_B_np[i:i + entropy_kernel_size, j:j + entropy_kernel_size]
            hist, _ = np.histogram(patch, bins=32, range=(0, 256))
            hist_normalized = hist / (entropy_kernel_size ** 2)
            entropy_val = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            entropy[i:i + entropy_kernel_size, j:j + entropy_kernel_size] = entropy_val

    # 平滑熵图以填充空隙并减少噪声
    entropy = cv2.GaussianBlur(entropy, (0, 0), 5)
    entropy_feature = cv2.normalize(entropy, None, 0, 1, cv2.NORM_MINMAX)

    # 自适应权重计算
    grad_strength = np.mean(gradient_feature)
    entropy_strength = np.mean(entropy_feature)
    total_strength = grad_strength + entropy_strength

    gradient_weight_adaptive = grad_strength / (total_strength + 1e-8)
    entropy_weight_adaptive = 1 - gradient_weight_adaptive

    # 合并两种特征（使用自适应权重）
    weight_map = gradient_weight_adaptive * gradient_feature + entropy_weight_adaptive * entropy_feature
    weight_map = cv2.normalize(weight_map, None, 0, 1, cv2.NORM_MINMAX)

    # 处理可能的无效值
    weight_map = np.nan_to_num(weight_map, nan=0)
    weight_map = np.maximum(0, weight_map)  # 确保没有负值

    # 如果输入是tensor，返回tensor
    if is_tensor:
        weight_map = torch.from_numpy(weight_map).float()
        if device:
            weight_map = weight_map.to(device)
        # 根据原始tensor形状调整输出
        if img_B.dim() == 4:
            weight_map = weight_map.unsqueeze(0).unsqueeze(0)
        elif img_B.dim() == 3:
            weight_map = weight_map.unsqueeze(0)

    return weight_map


def enhance_image(img_A, detail_weight_map, brightness_weight_map=None, detail_strength=0.8,
                  brightness_strength=0.6):
    """
    使用权重图智能增强图像A的细节和亮度

    参数:
        img_A: 要增强的输入图像 (numpy数组或tensor)
        detail_weight_map: 细节权重图，指示哪些区域需要增强细节
        brightness_weight_map: 亮度权重图，指示哪些区域需要增强亮度
        detail_strength: 细节增强强度
        brightness_strength: 亮度增强强度
        sharpness: 额外锐化强度

    返回:
        增强后的图像
    """
    # 将tensor转换为numpy处理
    is_tensor = isinstance(img_A, torch.Tensor)
    original_shape = None
    device = None

    if is_tensor:
        device = img_A.device
        original_shape = img_A.shape
        # 处理不同形状的tensor
        if img_A.dim() == 4:  # [B,C,H,W]
            img_A_np = img_A[0, 0].detach().cpu().numpy()
        elif img_A.dim() == 3:  # [C,H,W]
            img_A_np = img_A[0].detach().cpu().numpy()
        else:  # [H,W]
            img_A_np = img_A.detach().cpu().numpy()

        if isinstance(detail_weight_map, torch.Tensor):
            if detail_weight_map.dim() == 4:
                detail_weight_map = detail_weight_map[0, 0].detach().cpu().numpy()
            elif detail_weight_map.dim() == 3:
                detail_weight_map = detail_weight_map[0].detach().cpu().numpy()
            else:
                detail_weight_map = detail_weight_map.detach().cpu().numpy()

        if brightness_weight_map is not None and isinstance(brightness_weight_map, torch.Tensor):
            if brightness_weight_map.dim() == 4:
                brightness_weight_map = brightness_weight_map[0, 0].detach().cpu().numpy()
            elif brightness_weight_map.dim() == 3:
                brightness_weight_map = brightness_weight_map[0].detach().cpu().numpy()
            else:
                brightness_weight_map = brightness_weight_map.detach().cpu().numpy()
    else:
        img_A_np = img_A

    # 确保输入是灰度图
    if len(img_A_np.shape) > 2:
        img_A_np = cv2.cvtColor(img_A_np, cv2.COLOR_RGB2GRAY)

    # 归一化到[0,255]范围
    if img_A_np.max() <= 1.0:
        img_A_np = img_A_np * 255.0

    # 转换为浮点数进行计算
    img_A_float = img_A_np.astype(np.float32)

    # 创建多尺度细节提取 - 使用高斯金字塔
    gaussian_pyramid = [img_A_float]
    laplacian_pyramid = []

    # 生成3层高斯金字塔
    for i in range(3):
        gaussian_pyramid.append(cv2.GaussianBlur(gaussian_pyramid[-1], (0, 0), 1.5))

    # 生成拉普拉斯金字塔（细节层）
    for i in range(3):
        laplacian_pyramid.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])

    # 根据权重图增强细节
    enhanced_detail = np.zeros_like(img_A_float)
    for i, detail in enumerate(laplacian_pyramid):
        scale_weight = detail_strength * (1.0 - i * 0.2)
        enhanced_detail += detail * (1.0 + scale_weight * detail_weight_map)

    # 添加回基础层
    enhanced = gaussian_pyramid[-1] + enhanced_detail

    # 应用亮度增强
    if brightness_weight_map is not None:
        enhanced = enhanced + brightness_strength * brightness_weight_map * (255.0 - enhanced)
    # 处理可能的无效值并裁剪范围
    enhanced = np.nan_to_num(enhanced, nan=0, posinf=255, neginf=0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # 如果输入是tensor，返回tensor
    if is_tensor:
        enhanced = torch.from_numpy(enhanced).float()
        if device:
            enhanced = enhanced.to(device)

        # 如果原来是归一化的，输出也归一化
        if original_shape:
            if img_A.max() <= 1.0:
                enhanced = enhanced / 255.0

            # 恢复原始tensor形状
            if len(original_shape) == 4:  # [B,C,H,W]
                enhanced = enhanced.unsqueeze(0).unsqueeze(0)
            elif len(original_shape) == 3:  # [C,H,W]
                enhanced = enhanced.unsqueeze(0)

    return enhanced

