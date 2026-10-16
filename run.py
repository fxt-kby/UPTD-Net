# Code for the paper:
# UPTD-Net: Unsupervised Polarization-Guided Two-Stage Decoupling for Underwater Low-Light Image Restoration
# Optics & Laser Technology, vol. 204, article 116479, 2026
# https://doi.org/10.1016/j.optlastec.2026.116479

import cv2
from utils import *
from loss import *
from siren import *
import os
import numpy as np
from PIL import Image
from options import opt
import glob
import torch
from torch.amp import autocast, GradScaler

if __name__ == "__main__":
    if not os.path.exists(opt.datasets):
        print('input folder: {} does not exist'.format(opt.datasets))
        exit()
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    if not os.path.exists(os.path.join(opt.output_folder, 'd')):
        os.makedirs(os.path.join(opt.output_folder, 'd'))
    if not os.path.exists(os.path.join(opt.output_folder, 'b')):
        os.makedirs(os.path.join(opt.output_folder, 'b'))
    if not os.path.exists(os.path.join(opt.output_folder, 'd_enhanced')):
        os.makedirs(os.path.join(opt.output_folder, 'd_enhanced'))


    def sort_key_hazy(path):
        return int(os.path.basename(path).split('_')[1].split('.')[0])
    def sort_key_dop(path):
        return int(os.path.basename(path).split('.')[0])


    hazy_img_paths = sorted(glob.glob(opt.datasets + '/*.bmp'), key=sort_key_hazy)
    dop_img_paths = sorted(glob.glob(opt.datasets2 + '/*.bmp'), key=sort_key_dop)

    D_net = Net(32).cuda()
    B_net = Net2(16).cuda()
    img_siren = INF(patch_dim=opt.window ** 2, num_layers=4, hidden_dim=256, add_layer=2,
                    initial_dop_weight=1).cuda()

    l_TV = L_TV()
    mse_loss = torch.nn.MSELoss()

    scaler_D = GradScaler()
    scaler_B = GradScaler()
    scaler_I = GradScaler()
    print(' > running')
    for hazy_img_path, dop_img_path in zip(hazy_img_paths, dop_img_paths):
        print(f"Processing: {os.path.basename(hazy_img_path)}")

        img_rgb = get_image(hazy_img_path)  # 原始图像
        dop = get_image(dop_img_path)  # 偏振度图

        optimizer_D = torch.optim.Adam(D_net.parameters(), lr=opt.learning_rate)
        optimizer_B = torch.optim.Adam(B_net.parameters(), lr=opt.learning_rate2)
        optimizer_I = torch.optim.Adam(img_siren.parameters(), lr=opt.learning_rate3, betas=(0.9, 0.999),
                                       weight_decay=3e-4)

        height, width = img_rgb.shape[2], img_rgb.shape[3]
        coords = get_coords(height, width)

        lowpass = None
        D_out = None
        B_out = None
        # ============= 第一阶段：训练D_net网络与B_net网络 =============
        print("Stage 1: Training D_net&B_net")
        for epoch in range(301):
            D_net.train()
            B_net.train()
            optimizer_D.zero_grad()
            optimizer_B.zero_grad()

            with autocast(device_type='cuda'):
                lowpass = gaussian_low_pass_filter(img_rgb)
                D_out = D_net(img_rgb)
                B_out = B_net(img_rgb)
                normalized_img_rgb = img_rgb / img_rgb.max()

                loss1 = mse_loss(B_out, lowpass)
                loss2 = mse_loss(D_out, img_rgb)
                grad_loss = gradient_loss(D_out, normalized_img_rgb)
                reconstruction_loss = mse_loss((B_out + D_out), img_rgb)

                reconstruction_loss_D = mse_loss((B_out.detach() + D_out), img_rgb)
                total_lossD = 1 * mse_loss(D_out, img_rgb) + gradient_loss(D_out,
                                                                           normalized_img_rgb) + reconstruction_loss_D
                reconstruction_loss_B = mse_loss((B_out + D_out.detach()), img_rgb)
                total_lossB = 5 * mse_loss(B_out, lowpass) + 1 * reconstruction_loss_B + l_TV(B_out)

            scaler_D.scale(total_lossD).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            scaler_B.scale(total_lossB).backward()
            scaler_B.step(optimizer_B)
            scaler_B.update()

            if epoch % 50 == 0:
                print(f'B_net Epoch {epoch:05d}  Loss: {total_lossB.item():.6f}')
                print(f'D_net Epoch {epoch:05d}  Loss: {total_lossD.item():.6f}')

        B_out = B_out
        D_out = D_out
        normalized_D_out = D_out / D_out.max()

        img_name = str(os.path.basename(hazy_img_path))
        eme_img_path = os.path.join(str(opt.datasets3), img_name.replace('.bmp', '.bmp'))
        # eme_img_path = os.path.join(str(opt.datasets3), img_name.replace('.bmp', '.bmp'))

        if os.path.exists(eme_img_path):
            img_B = cv2.imread(eme_img_path, 0)  # 读取为灰度图像
            img_B = torch.from_numpy(img_B).float().unsqueeze(0).unsqueeze(0).cuda() / 255.0
        else:
            print(f"Error: EME reference image not found: {eme_img_path}")
            continue

        detail_weight_map = create_advanced_weight_map(img_B)
        brightness_weight_map = create_brightness_weight_map(img_B, threshold_factor=1, smoothness=0)

        D_out_en = enhance_image(
            D_out,
            detail_weight_map,
            brightness_weight_map,
            detail_strength=1,
            brightness_strength=0.8,
        )

        # ============= 第二阶段 =============
        print("Stage 2: Training RI-Denet")
        img_v_fixed_lr = None
        illu_lr = None

        for epoch in range(opt.epochs):
            img_siren.train()
            optimizer_I.zero_grad()

            with autocast(device_type='cuda'):
                D_en_detached = D_out_en.detach()
                patches = get_patches(D_en_detached, opt.window)
                illu_res_lr = img_siren(patches, coords)
                illu_res_lr = illu_res_lr.view(1, 1, opt.down_size, opt.down_size)
                illu_lr = illu_res_lr + D_en_detached
                img_v_fixed_lr = D_en_detached / (illu_lr + 1e-5)

                loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - D_en_detached, 2)))
                loss_tv = l_TV(illu_lr)
                dop_weight = img_siren.get_dop_weight()
                t = (1 - (dop * dop_weight))

                It = t
                It = It * 0.5
                loss_exp = mse_loss(illu_lr, It)
                loss_sparsity = torch.mean(torch.abs(img_v_fixed_lr))
                total_loss_2 = loss_spa * opt.alpha + loss_tv * opt.beta + loss_exp * opt.gamma + loss_sparsity * opt.delta

            scaler_I.scale(total_loss_2).backward()
            scaler_I.step(optimizer_I)
            scaler_I.update()

            if epoch % 10 == 0:
                print(
                    f'img_siren Epoch {epoch:05d}  Loss_spa: {loss_spa.item():.6f} loss_tv: {loss_tv.item():.6f} loss_exp: {loss_exp.item():.6f}  loss_sparsity: {loss_sparsity.item():.6f}')
        final_dop_weight = img_siren.get_dop_weight()
        img_v_numpy = img_v_fixed_lr[0, 0].detach().cpu().numpy()
        p_low, p_high = np.percentile(img_v_numpy, (0, 100))
        img_v_stretched = np.clip((img_v_numpy - p_low) / (p_high - p_low), 0, 1)

        Image.fromarray(
            ((torch.movedim(D_out, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'd', os.path.basename(hazy_img_path)))

        Image.fromarray(
            ((torch.movedim(D_out_en, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'd_enhanced', os.path.basename(hazy_img_path)))

        Image.fromarray(
            ((torch.movedim(B_out, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'b', os.path.basename(hazy_img_path)))

        Image.fromarray(
            (img_v_stretched * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, os.path.basename(hazy_img_path)))
        torch.cuda.empty_cache()

    print(' > reconstruction done')
