#  加入细节和亮度补偿
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

    # 创建保存输出图像的子文件夹
    if not os.path.exists(os.path.join(opt.output_folder, 'd')):
        os.makedirs(os.path.join(opt.output_folder, 'd'))
    if not os.path.exists(os.path.join(opt.output_folder, 'lowpass')):
        os.makedirs(os.path.join(opt.output_folder, 'lowpass'))
    if not os.path.exists(os.path.join(opt.output_folder, 'b')):
        os.makedirs(os.path.join(opt.output_folder, 'b'))
    if not os.path.exists(os.path.join(opt.output_folder, 'illu')):
        os.makedirs(os.path.join(opt.output_folder, 'illu'))
    # 创建保存增强图像的文件夹
    if not os.path.exists(os.path.join(opt.output_folder, 'd_enhanced')):
        os.makedirs(os.path.join(opt.output_folder, 'd_enhanced'))


    # 自定义排序键，按文件名排序
    def sort_key_hazy(path):
        return int(os.path.basename(path).split('_')[1].split('.')[0])


    def sort_key_dop(path):
        return int(os.path.basename(path).split('.')[0])


    hazy_img_paths = sorted(glob.glob(opt.datasets + '/*.bmp'), key=sort_key_hazy)
    dop_img_paths = sorted(glob.glob(opt.datasets2 + '/*.bmp'), key=sort_key_dop)

    # 定义网络模型
    D_net = Net(32).cuda()
    B_net = Net2(16).cuda()
    img_siren = INF(patch_dim=opt.window ** 2, num_layers=4, hidden_dim=256, add_layer=2,
                    initial_dop_weight=1).cuda()  # ############# initial_dop_weight=1.001)

    # 定义损失函数
    l_TV = L_TV()
    mse_loss = torch.nn.MSELoss()

    # 初始化GradScaler
    scaler_D = GradScaler()
    scaler_B = GradScaler()
    scaler_I = GradScaler()
    # 遍历 input 中的每一个图像文件
    print(' > running')
    for hazy_img_path, dop_img_path in zip(hazy_img_paths, dop_img_paths):
        print(f"Processing: {os.path.basename(hazy_img_path)}")

        img_rgb = get_image(hazy_img_path)  # 原始图像
        dop = get_image(dop_img_path)  # 偏振度图

        # 重新初始化优化器
        optimizer_D = torch.optim.Adam(D_net.parameters(), lr=opt.learning_rate)
        optimizer_B = torch.optim.Adam(B_net.parameters(), lr=opt.learning_rate2)
        optimizer_I = torch.optim.Adam(img_siren.parameters(), lr=opt.learning_rate3, betas=(0.9, 0.999),
                                       weight_decay=3e-4)

        height, width = img_rgb.shape[2], img_rgb.shape[3]  # 假设img_rgb的形状为[B,C,H,W]
        coords = get_coords(height, width)  # 构建与图像尺寸一致的网格

        lowpass = None
        D_out = None
        B_out = None
        # ============= 第一阶段：训练D_net网络与B_net网络500次 =============
        print("Stage 1: Training D_net&B_net for 500 epochs")
        for epoch in range(301):  # 固定训练500次
            D_net.train()
            B_net.train()
            optimizer_D.zero_grad()
            optimizer_B.zero_grad()

            with autocast(device_type='cuda'):
                lowpass = gaussian_low_pass_filter(img_rgb)
                D_out = D_net(img_rgb)
                B_out = B_net(img_rgb)  # 背向散射光
                normalized_img_rgb = img_rgb / img_rgb.max()

                # 只计算第一阶段的损失
                loss1 = mse_loss(B_out, lowpass)  # 背向散射光与低通滤波相似
                loss2 = mse_loss(D_out, img_rgb)  # 去雾图与原始图像相似
                grad_loss = gradient_loss(D_out, normalized_img_rgb)  # 去雾图与原始图像归一化图相似
                reconstruction_loss = mse_loss((B_out + D_out), img_rgb)

                # 分离计算图
                reconstruction_loss_D = mse_loss((B_out.detach() + D_out), img_rgb)
                total_lossD = 1 * mse_loss(D_out, img_rgb) + gradient_loss(D_out,
                                                                           normalized_img_rgb) + reconstruction_loss_D
                # 默认1 1 1  auv_move 5 1 1 chidi_crop 5 1 1 auv511  auvvideo 5 1 1
                reconstruction_loss_B = mse_loss((B_out + D_out.detach()), img_rgb)
                total_lossB = 5 * mse_loss(B_out, lowpass) + 1 * reconstruction_loss_B + l_TV(B_out)
                # 默认5 1 1auv_move 10 1 1

            # 只更新D_net网络
            scaler_D.scale(total_lossD).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            # 只更新B_net网络
            scaler_B.scale(total_lossB).backward()
            scaler_B.step(optimizer_B)
            scaler_B.update()

            if epoch % 50 == 0:
                print(f'B_net Epoch {epoch:05d}  Loss: {total_lossB.item():.6f}')
                print(f'D_net Epoch {epoch:05d}  Loss: {total_lossD.item():.6f}')

        # 第一阶段训练完成，保存D_out的结果
        B_out = B_out
        D_out = D_out
        normalized_D_out = D_out / D_out.max()

        # ============= 应用图像增强 =============
        img_name = str(os.path.basename(hazy_img_path))
        eme_img_path = os.path.join(str(opt.datasets3), img_name.replace('.bmp', '.bmp'))
        # eme_img_path = os.path.join(str(opt.datasets3), img_name.replace('.bmp', '.bmp'))

        if os.path.exists(eme_img_path):
            # 读取参考图像并调整大小为512×512
            img_B = cv2.imread(eme_img_path, 0)  # 读取为灰度图像
            # img_B = img_B[100:1024, 100:1024]   # 裁剪成方形 [:, :1024]        [100:1024, 100:1024]#################################
            # img_B = cv2.resize(img_B, (512, 512))  # 调整大小为512×512
            img_B = torch.from_numpy(img_B).float().unsqueeze(0).unsqueeze(0).cuda() / 255.0
        else:
            print(f"Error: EME reference image not found: {eme_img_path}")
            continue  # 跳过当前图像处理

        # 创建细节和亮度权重图
        detail_weight_map = create_advanced_weight_map(img_B)  # #########
        brightness_weight_map = create_brightness_weight_map(img_B, threshold_factor=1, smoothness=0)

        # 对D_out进行细节和亮度增强
        D_out_en = enhance_image(
            D_out,
            detail_weight_map,
            brightness_weight_map,
            detail_strength=1,
            brightness_strength=0.8,
        )

        # 计算并输出EME评价指标
        eme_original = calculate_eme(img_rgb)
        eme_D_out = calculate_eme(D_out)
        eme_D_out_en = calculate_eme(D_out_en)

        print(f"EME评价指标:")
        print(f"- 原始图像(img_rgb): {eme_original:.4f}")
        print(f"- 去雾图像(D_out): {eme_D_out:.4f} (增加: {(eme_D_out - eme_original) / eme_original * 100:.2f}%)")
        print(
            f"- 增强图像(D_out_en): {eme_D_out_en:.4f} (增加: {(eme_D_out_en - eme_original) / eme_original * 100:.2f}%)")

        # ============= 第二阶段：训练img_siren网络 =============
        print("Stage 2: Training img_siren network")
        img_v_fixed_lr = None
        illu_lr = None
        final_output = None

        # 使用增强后的D_out_en替代D_out作为输入
        for epoch in range(opt.epochs):
            img_siren.train()
            optimizer_I.zero_grad()

            with autocast(device_type='cuda'):
                # 使用增强后的图像D_lr_en代替D_lr
                # D_lr_en_detached = D_lr_en.detach()
                D_en_detached = D_out_en.detach()
                patches = get_patches(D_en_detached, opt.window)
                illu_res_lr = img_siren(patches, coords)  # 光照和去雾图的残差
                illu_res_lr = illu_res_lr.view(1, 1, opt.down_size, opt.down_size)
                illu_lr = illu_res_lr + D_en_detached  # 光照
                img_v_fixed_lr = D_en_detached / (illu_lr + 1e-5)  # 去雾图除以光照，反射图

                final_output = img_v_fixed_lr  # 保存最终输出

                loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - D_en_detached, 2)))
                loss_tv = l_TV(illu_lr)

                # ============= 关键修改：使用可学习的DOP权重 =============
                dop_weight = img_siren.get_dop_weight()  # 获取当前的DOP权重
                t = (1 - (dop * dop_weight))  # 使用可学习参数替代固定的1.1

                It = t  # 认为是理想光照0.4，越小越亮
                # It = It / torch.max(It)  # 归一化
                It = It * 0.5  # 普通为0.5 曝光3000*0.5 frame7*0.6  auv_move 0.7  chidi0.6 auvvideo0.6
                loss_exp = mse_loss(illu_lr, It)
                loss_sparsity = torch.mean(torch.abs(img_v_fixed_lr))
                # 计算第二阶段的损失
                total_loss_2 = loss_spa * opt.alpha + loss_tv * opt.beta + loss_exp * opt.gamma + loss_sparsity * opt.delta

            # 更新img_siren网络
            scaler_I.scale(total_loss_2).backward()
            scaler_I.step(optimizer_I)
            scaler_I.update()

            if epoch % 10 == 0:
                print(
                    f'img_siren Epoch {epoch:05d}  Loss_spa: {loss_spa.item():.6f} loss_tv: {loss_tv.item():.6f} loss_exp: {loss_exp.item():.6f}  loss_sparsity: {loss_sparsity.item():.6f}')
        # 在第二阶段训练结束后输出最终的可学习DOP权重
        final_dop_weight = img_siren.get_dop_weight()
        print(f"Final DOP weight for {os.path.basename(hazy_img_path)}: {final_dop_weight.item():.6f}")

        # img_v_stretched = (img_v_fixed_lr - torch.min(img_v_fixed_lr))/ (
        #             torch.max(img_v_fixed_lr) - torch.min(img_v_fixed_lr) + 1e-8)

        # ========== 新增：百分位拉伸处理 ==========
        img_v_numpy = img_v_fixed_lr[0, 0].detach().cpu().numpy()
        p_low, p_high = np.percentile(img_v_numpy, (0, 100))
        img_v_stretched = np.clip((img_v_numpy - p_low) / (p_high - p_low), 0, 1)

        # ========== 百分位拉伸处理结束 ==========

        # 计算最终输出的EME值
        eme_final = calculate_eme(final_output)
        print(
            f"- 最终输出(final_output): {eme_final:.4f} (增加: {(eme_final - eme_original) / eme_original * 100:.2f}%)")

        # 保存处理后的图像
        Image.fromarray(
            ((torch.movedim(D_out, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'd', os.path.basename(hazy_img_path)))

        # 保存增强后的D_out_en
        Image.fromarray(
            ((torch.movedim(D_out_en, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'd_enhanced', os.path.basename(hazy_img_path)))

        Image.fromarray(
            ((torch.movedim(B_out, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'b', os.path.basename(hazy_img_path)))

        # Image.fromarray(
        #     (img_v_fixed_lr[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        # ).save(os.path.join(opt.output_folder, os.path.basename(hazy_img_path)))
        # 保存最终图像
        Image.fromarray(
            (img_v_stretched * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, os.path.basename(hazy_img_path)))

        Image.fromarray(
            ((torch.movedim(lowpass, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, 'lowpass', os.path.basename(hazy_img_path)))

        # 保存中间结果：illu_lr
        if illu_lr is not None:
            Image.fromarray(
                ((torch.movedim(illu_lr, 1, -1)[0]).squeeze(2).detach().cpu().numpy() * 255).astype(np.uint8)
            ).save(os.path.join(opt.output_folder, 'illu', os.path.basename(hazy_img_path)))

        # 记录所有EME值到日志文件
        with open(os.path.join(opt.output_folder, 'eme_log.txt'), 'a') as f:
            f.write(f"Image: {os.path.basename(hazy_img_path)}\n")
            f.write(f"- 原始图像(img_rgb): {eme_original:.4f}\n")
            f.write(
                f"- 去雾图像(D_out): {eme_D_out:.4f} (增加: {(eme_D_out - eme_original) / eme_original * 100:.2f}%)\n")
            f.write(
                f"- 增强图像(D_out_en): {eme_D_out_en:.4f} (增加: {(eme_D_out_en - eme_original) / eme_original * 100:.2f}%)\n")
            f.write(
                f"- 最终输出(final_output): {eme_final:.4f} (增加: {(eme_final - eme_original) / eme_original * 100:.2f}%)\n")
            f.write(
                f"- 可学习参数 c (DOP weight, clamped): {final_dop_weight.item():.6f}\n")
            f.write("------------------------\n")

        # 清理GPU缓存
        torch.cuda.empty_cache()

    print(' > reconstruction done')
