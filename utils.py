import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from filter import FastGuidedFilter


def get_image(path, device='cuda'):
    """
    读取图像并返回 4D Tensor (1, 1, H, W)，强制灰度。
    """
    img = Image.open(path).convert('L')
    arr = np.array(img).astype('float32') / 255.0
    arr = arr[None, ...]
    tensor = torch.from_numpy(arr).unsqueeze(0)
    if device == 'cuda' and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def get_v_component(img_hsv):
    """
    Assumes (1,3,H,W) HSV image.
    """
    return img_hsv[:, -1].unsqueeze(0)


def replace_v_component(img_hsv, v_new):
    """
    Replaces the V component of a HSV image (1,3,H,W).
    """
    img_hsv[:, -1] = v_new
    return img_hsv


def interpolate_image(img, H, W):
    """
    Reshapes the image based on new resolution.
    """
    return F.interpolate(img, size=(H, W))


def get_coords(H, W):
    """
    Creates a coordinates grid for INF.
    """
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float().cuda()
    return coords


def get_patches(img, KERNEL_SIZE):
    """
    Creates a tensor where the channel contains patch information.
    """
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad = nn.ReflectionPad2d(KERNEL_SIZE // 2)
    im_padded = pad(img)
    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)

    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr, y_lr, x_hr, r=2):
    """
    Applies the guided filter to upscale the predicted image.
    """
    x_lr = torch.nan_to_num(x_lr, nan=0.0, posinf=0.99, neginf=0.01)
    y_lr = torch.nan_to_num(y_lr, nan=0.0, posinf=0.99, neginf=0.01)
    x_hr = torch.nan_to_num(x_hr, nan=0.0, posinf=0.99, neginf=0.01)
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr


def np_to_torch(img_np):
    return torch.from_numpy(img_np).unsqueeze(0)


def prepare_image(file_path):
    image = Image.open(file_path).convert('L')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def gaussian_low_pass_filter(image_tensor, sigma=3):
    """
    对图像张量进行高斯低通滤波
    :param image_tensor: 输入图像张量 [batch_size, channels, height, width]
    :param sigma: 高斯核的标准差
    :return: 滤波后的图像张量 默认sigma=3

    """
    if len(image_tensor.shape) != 4:
        raise ValueError("Expected 4D tensor [batch_size, channels, height, width]")

    device = image_tensor.device
    B, C, H, W = image_tensor.shape
    x = torch.linspace(-W // 2, W // 2, W, device=device)
    y = torch.linspace(-H // 2, H // 2, H, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    D = torch.sqrt(X ** 2 + Y ** 2)
    H = torch.exp(-(D ** 2) / (2 * sigma ** 2))
    H = H.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
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
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量")

    device = tensor.device

    tensor_np = tensor.detach().cpu().numpy()

    filtered = np.zeros_like(tensor_np)

    for b in range(tensor_np.shape[0]):
        for c in range(tensor_np.shape[1]):

            filtered[b, c] = cv2.blur(tensor_np[b, c].astype(np.float32), (kernel_size, kernel_size))

    filtered_tensor = torch.from_numpy(filtered).to(device)

    return filtered_tensor



def calculate_eme(img, block_size=8):
    """
    计算图像的EME (Enhancement Measure Evaluation) 值

    参数:
        img: 输入图像 (可以是tensor或numpy数组)
        block_size: 分块大小

    返回:
        EME值
    """
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.detach().cpu()
        if img.dim() == 4:  # [B,C,H,W]
            img = img[0, 0].numpy()
        elif img.dim() == 3:  # [C,H,W]
            img = img[0].numpy()
        else:
            img = img.numpy()

    if len(img.shape) > 2:
        if img.shape[2] == 3 or img.shape[0] == 3:  # HWC or CHW
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.max() <= 1.0:
        img = img * 255.0

    img = np.maximum(img.astype(np.float32), 0.001)

    h, w = img.shape
    k1 = h // block_size
    k2 = w // block_size

    sum_eme = 0

    for i in range(k1):
        for j in range(k2):
            block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            block_max = np.max(block)
            block_min = np.min(block)

            if 0 < block_min < block_max:
                sum_eme += 20.0 * np.log10(block_max / block_min)

    eme = sum_eme / (k1 * k2) if k1 * k2 > 0 else 0
    return eme


def create_brightness_weight_map(img_B, threshold_factor=1, smoothness=0):
    """
    创建亮度权重图，只在高于平均亮度的区域有权重

    参数:
        img_B: 输入图像 (numpy数组或tensor)
        threshold_factor: 亮度阈值因子，默认为1.0表示使用平均亮度作为阈值
        smoothness: 权重图平滑因子

    返回:
        亮度权重图，高于平均亮度的区域有权重，低于的区域权重为0
    """
    is_tensor = isinstance(img_B, torch.Tensor)
    device = None

    if is_tensor:
        device = img_B.device
        if img_B.dim() == 4:  # [B,C,H,W]
            img_B_np = img_B[0, 0].detach().cpu().numpy()
        elif img_B.dim() == 3:  # [C,H,W]
            img_B_np = img_B[0].detach().cpu().numpy()
        else:  # [H,W]
            img_B_np = img_B.detach().cpu().numpy()
    else:
        img_B_np = img_B

    if len(img_B_np.shape) > 2:
        img_B_np = cv2.cvtColor(img_B_np, cv2.COLOR_RGB2GRAY)

    if img_B_np.max() <= 1.0:
        img_B_np = img_B_np * 255.0

    mean_brightness = np.mean(img_B_np)
    threshold = mean_brightness * threshold_factor

    brightness_map = np.zeros_like(img_B_np, dtype=np.float32)

    mask1 = img_B_np > threshold
    brightness_map[mask1] = (img_B_np[mask1] - threshold)/ (255.0 - threshold)
    mask2 = img_B_np < threshold
    brightness_map[mask2] = ((img_B_np[mask2]-threshold) / (255.0 - threshold)) * 0.4

    brightness_map = cv2.normalize(brightness_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if smoothness > 0:
        brightness_map = cv2.GaussianBlur(brightness_map, (0, 0), smoothness)

    brightness_map = cv2.normalize(brightness_map, None, 0, 1, cv2.NORM_MINMAX)

    if is_tensor:
        brightness_map = torch.from_numpy(brightness_map).float()
        if device:
            brightness_map = brightness_map.to(device)
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
    is_tensor = isinstance(img_B, torch.Tensor)
    device = None

    if is_tensor:
        device = img_B.device
        if img_B.dim() == 4:  # [B,C,H,W]
            img_B_np = img_B[0, 0].detach().cpu().numpy()
        elif img_B.dim() == 3:  # [C,H,W]
            img_B_np = img_B[0].detach().cpu().numpy()
        else:  # [H,W]
            img_B_np = img_B.detach().cpu().numpy()
    else:
        img_B_np = img_B

    if len(img_B_np.shape) > 2:
        img_B_np = cv2.cvtColor(img_B_np, cv2.COLOR_RGB2GRAY)

    if img_B_np.max() <= 1.0:
        img_B_np = img_B_np * 255.0

    img_B_float = img_B_np.astype(np.float32)

    grad_x = cv2.Sobel(img_B_float, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img_B_float, cv2.CV_32F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_feature = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    entropy_kernel_size = 5  # 熵计算的窗口大小
    entropy = np.zeros_like(img_B_float)
    h, w = img_B_np.shape

    for i in range(0, h - entropy_kernel_size, 2):
        for j in range(0, w - entropy_kernel_size, 2):
            patch = img_B_np[i:i + entropy_kernel_size, j:j + entropy_kernel_size]
            hist, _ = np.histogram(patch, bins=32, range=(0, 256))
            hist_normalized = hist / (entropy_kernel_size ** 2)
            entropy_val = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            entropy[i:i + entropy_kernel_size, j:j + entropy_kernel_size] = entropy_val

    entropy = cv2.GaussianBlur(entropy, (0, 0), 5)
    entropy_feature = cv2.normalize(entropy, None, 0, 1, cv2.NORM_MINMAX)

    grad_strength = np.mean(gradient_feature)
    entropy_strength = np.mean(entropy_feature)
    total_strength = grad_strength + entropy_strength

    gradient_weight_adaptive = grad_strength / (total_strength + 1e-8)
    entropy_weight_adaptive = 1 - gradient_weight_adaptive

    weight_map = gradient_weight_adaptive * gradient_feature + entropy_weight_adaptive * entropy_feature
    weight_map = cv2.normalize(weight_map, None, 0, 1, cv2.NORM_MINMAX)

    weight_map = np.nan_to_num(weight_map, nan=0)
    weight_map = np.maximum(0, weight_map)  # 确保没有负值

    if is_tensor:
        weight_map = torch.from_numpy(weight_map).float()
        if device:
            weight_map = weight_map.to(device)
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
    #
    is_tensor = isinstance(img_A, torch.Tensor)
    original_shape = None
    device = None

    if is_tensor:
        device = img_A.device
        original_shape = img_A.shape
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

    if len(img_A_np.shape) > 2:
        img_A_np = cv2.cvtColor(img_A_np, cv2.COLOR_RGB2GRAY)

    if img_A_np.max() <= 1.0:
        img_A_np = img_A_np * 255.0

    img_A_float = img_A_np.astype(np.float32)

    gaussian_pyramid = [img_A_float]
    laplacian_pyramid = []

    for i in range(3):
        gaussian_pyramid.append(cv2.GaussianBlur(gaussian_pyramid[-1], (0, 0), 1.5))

    for i in range(3):
        laplacian_pyramid.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])

    enhanced_detail = np.zeros_like(img_A_float)
    for i, detail in enumerate(laplacian_pyramid):
        scale_weight = detail_strength * (1.0 - i * 0.2)
        enhanced_detail += detail * (1.0 + scale_weight * detail_weight_map)

    enhanced = gaussian_pyramid[-1] + enhanced_detail

    if brightness_weight_map is not None:
        enhanced = enhanced + brightness_strength * brightness_weight_map * (255.0 - enhanced)
    enhanced = np.nan_to_num(enhanced, nan=0, posinf=255, neginf=0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    if is_tensor:
        enhanced = torch.from_numpy(enhanced).float()
        if device:
            enhanced = enhanced.to(device)

        if original_shape:
            if img_A.max() <= 1.0:
                enhanced = enhanced / 255.0

            if len(original_shape) == 4:  # [B,C,H,W]
                enhanced = enhanced.unsqueeze(0).unsqueeze(0)
            elif len(original_shape) == 3:  # [C,H,W]
                enhanced = enhanced.unsqueeze(0)

    return enhanced

