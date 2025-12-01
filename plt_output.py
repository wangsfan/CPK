import os, torch
import matplotlib.pyplot as plt
import numpy as np


def _get_palette(nc=21):
    # Fixed palette length >= nc
    colors = np.array([
        [0, 0, 0],          # 0: 黑色（背景）
        [255, 0, 0],        # 1: 红色
        [0, 255, 0],        # 2: 绿色
        [0, 0, 255],        # 3: 蓝色
        [255, 255, 0],      # 4: 黄色
        [255, 0, 255],      # 5: 紫色
        [0, 255, 255],      # 6: 青色
        [128, 0, 0],        # 7: 深红色
        [0, 128, 0],        # 8: 深绿色
        [0, 0, 128],        # 9: 深蓝色
        [128, 128, 0],      # 10: 橄榄色
        [128, 0, 128],      # 11: 深紫色
        [0, 128, 128],      # 12: 深青色
        [64, 0, 0],         # 13: 更深的红色
        [0, 64, 0],         # 14: 更深的绿色
        [0, 0, 64],         # 15: 更深的蓝色
        [64, 64, 0],        # 16: 更深的黄色
        [64, 0, 64],        # 17: 更深的紫色
        [0, 64, 64],        # 18: 更深的青色
        [192, 192, 192],    # 19: 灰色
        [128, 128, 0],      # 20: 黄绿色
        [0, 128, 128],      # 21: 水蓝色
        [255, 128, 128],    # 22: 浅红色
        [128, 255, 128],    # 23: 浅绿色
        [128, 128, 255],    # 24: 浅蓝色
        [255, 255, 128],    # 25: 浅黄色
    ], dtype=np.uint8)
    if nc <= colors.shape[0]:
        return colors[:nc]
    # Repeat if more classes than palette entries
    reps = int(np.ceil(nc / colors.shape[0]))
    return np.tile(colors, (reps, 1))[:nc]


def _to_numpy_image(image_tensor):
    # image_tensor: CxHxW (tensor on cpu)
    img = image_tensor
    min_v = float(img.min())
    max_v = float(img.max())
    if max_v > min_v:
        img = (img - min_v) / (max_v - min_v)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    return img


def _colorize_mask(mask_hw, num_classes):
    palette = _get_palette(num_classes)
    h, w = mask_hw.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask_hw >= 0) & (mask_hw < num_classes)
    colored[valid] = palette[mask_hw[valid]]
    return colored


def _norm_heatmap(t_hw):
    arr = t_hw.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def viz_train_step(
    save_dir,
    images,
    labels,
    l1h,
    outputs_old=None,
    outputs=None,
    int_masks=None,
    int_masks_raw=None,
    pseudo_new_label=None,
    pseudo_label=None,
    int_masks_orig=None,
    int_masks_soft=None,
    pseudo_gt_seg=None,
    pseudo_gt_seg_lx=None,
    target_old=None,
    target=None,
    feat_old_body=None,
    feat_body=None,
    old_classes=0,
    tot_classes=21,
    cur_epoch=0,
    cur_step=0,
    max_rows=8,
):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        bs = images.shape[0]
        rows = min(bs, max_rows)

        cols = 2  # image + label
        show_outputs_old = outputs_old is not None
        show_outputs = outputs is not None
        show_int_masks = int_masks is not None
        show_int_masks_raw = int_masks_raw is not None
        show_pseudo_new = pseudo_new_label is not None
        show_pseudo = pseudo_label is not None
        show_feat_old = feat_old_body is not None
        show_feat = feat_body is not None

        cols += int(show_outputs_old)
        cols += int(show_outputs)
        cols += int(show_int_masks)
        cols += int(show_int_masks_raw)
        cols += int(show_pseudo_new)
        cols += int(show_pseudo)
        cols += int(show_feat_old)
        cols += int(show_feat)
        # newly requested tensors
        show_int_masks_orig = int_masks_orig is not None
        show_int_masks_soft = int_masks_soft is not None
        show_pseudo_gt_seg = pseudo_gt_seg is not None
        show_pseudo_gt_seg_lx = pseudo_gt_seg_lx is not None
        show_target_old = target_old is not None
        show_target = target is not None
        cols += int(show_int_masks_orig)
        cols += int(show_int_masks_soft)
        cols += int(show_pseudo_gt_seg)
        cols += int(show_pseudo_gt_seg_lx)
        cols += int(show_target_old)
        cols += int(show_target)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), constrained_layout=True)
        if rows == 1:
            axs = np.expand_dims(axs, 0)

        labels_np = labels.cpu().numpy()

        col_names = []
        for r in range(rows):
            c = 0
            img_np = _to_numpy_image(images[r].cpu())
            axs[r, c].imshow(img_np)
            axs[r, c].axis('off')
            if r == 0:
                col_names.append('image')
            c += 1

            lab_color = _colorize_mask(labels_np[r], tot_classes)
            axs[r, c].imshow(lab_color)
            axs[r, c].axis('off')
            if r == 0:
                col_names.append('label')
            c += 1

            if show_outputs_old:
                pred_old = torch.argmax(outputs_old[r], dim=0).cpu().numpy()
                pred_old_color = _colorize_mask(pred_old, tot_classes)
                axs[r, c].imshow(pred_old_color)
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('outputs_old(argmax)')
                c += 1

            if show_outputs:
                pred_new = torch.argmax(outputs[r], dim=0).cpu().numpy()
                pred_new_color = _colorize_mask(pred_new, tot_classes)
                axs[r, c].imshow(pred_new_color)
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('outputs(argmax)')
                c += 1

            if show_int_masks:
                # visualize argmax of soft masks
                pred_cam = torch.argmax(int_masks[r], dim=0).cpu().numpy()
                # zero is background, foreground classes in [1, ...]
                pred_cam_color = _colorize_mask(pred_cam, tot_classes)
                axs[r, c].imshow(pred_cam_color)
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('int_masks(argmax)')
                c += 1

            if show_int_masks_raw:
                pred_cam_raw = torch.argmax(int_masks_raw[r], dim=0).cpu().numpy()
                pred_cam_raw_color = _colorize_mask(pred_cam_raw, tot_classes)
                axs[r, c].imshow(pred_cam_raw_color)
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('int_masks_raw(argmax)')
                c += 1

            if show_pseudo_new:
                # shift new class indices by old_classes for visualization
                pred_pn = torch.argmax(pseudo_new_label[r], dim=0).cpu().numpy().astype(np.int32)
                pred_pn = np.where(pred_pn > 0, pred_pn + old_classes, 0)
                pred_pn_color = _colorize_mask(pred_pn, tot_classes)
                axs[r, c].imshow(pred_pn_color)
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('pseudo_new_label(argmax→global)')
                c += 1

            if show_pseudo:
                pred_ps = torch.argmax(pseudo_label[r], dim=0).cpu().numpy()
                pred_ps_color = _colorize_mask(pred_ps, tot_classes)
                axs[r, c].imshow(pred_ps_color)
                axs[r, c].axis('off')
                if r == 0:
                    # annotate activated new classes from l1h slice [:, old_classes-1:]
                    l1h_row = l1h[r].cpu().numpy()
                    act_local = np.where(l1h_row[max(old_classes-1, 0):] > 0.5)[0]
                    act_global = (act_local + max(old_classes-1, 0)).tolist()
                    # best-effort names for VOC
                    classes = {
                        0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
                        6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
                        13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa',
                        19: 'train', 20: 'tvmonitor'
                    }
                    names = [classes.get(i, f'C{i}') for i in act_global]
                    title = 'pseudo_label(argmax)\nnew: ' + (', '.join(names) if len(names) else 'none')
                    col_names.append(title)
                c += 1

            if show_feat_old:
                fmap = feat_old_body[r]  # CxHxW
                norm_map = torch.linalg.vector_norm(fmap, dim=0).cpu().numpy()
                axs[r, c].imshow(_norm_heatmap(norm_map), cmap='magma')
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('feat_old||·||')
                c += 1

            if show_feat:
                fmap = feat_body[r]
                norm_map = torch.linalg.vector_norm(fmap, dim=0).cpu().numpy()
                axs[r, c].imshow(_norm_heatmap(norm_map), cmap='magma')
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('feat||·||')
                c += 1

            # newly requested tensors
            if show_int_masks_orig:
                pred_imo = torch.argmax(int_masks_orig[r], dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_imo, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('int_masks_orig(argmax)')
                c += 1

            if show_int_masks_soft:
                pred_ims = torch.argmax(int_masks_soft[r], dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_ims, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('int_masks_soft(argmax)')
                c += 1

            if show_pseudo_gt_seg:
                # pseudo_gt_seg: B x C x HW -> reshape
                C = pseudo_gt_seg.shape[1]
                H, W = int_masks_soft.shape[-2:] if show_int_masks_soft else int_masks.shape[-2:]
                pg = pseudo_gt_seg[r].view(C, H, W)
                pred_pg = torch.argmax(pg, dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_pg, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('pseudo_gt_seg(argmax)')
                c += 1

            if show_pseudo_gt_seg_lx:
                pred_pgl = torch.argmax(pseudo_gt_seg_lx[r], dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_pgl, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('pseudo_gt_seg_lx(argmax)')
                c += 1

            if show_target_old:
                pred_to = torch.argmax(target_old[r], dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_to, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('target_old(argmax)')
                c += 1

            if show_target:
                pred_t = torch.argmax(target[r], dim=0).cpu().numpy()
                axs[r, c].imshow(_colorize_mask(pred_t, tot_classes))
                axs[r, c].axis('off')
                if r == 0:
                    col_names.append('target(argmax)')
                c += 1

        # set column titles
        for idx, name in enumerate(col_names):
            axs[0, idx].set_title(name)

        fig_path = os.path.join(save_dir, f'train_viz_{cur_epoch}_{cur_step}.png')
        plt.savefig(fig_path)
        plt.close(fig)
        print(f'[viz] saved {fig_path}')

def print1(save_dir, images, labels, l1h, sal_img, cur_epoch, cur_step):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    sal_images_np = sal_img.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    # outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示显著性图
        axs[i, 1].imshow(sal_images_np[i])
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'SalImg {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(label_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print2(save_dir, outputs, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels  # 先移动到CPU，再转换为numpy数组
    outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    # img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
    #                        indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(len(images)):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = outputs_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(output_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Output {i + 1}')
        # axs[i, 1].set_title(f'Output {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(label_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Label {i + 1}')
        # axs[i, 2].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print3(save_dir, outputs, new_class_pseudo, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(new_class_pseudo, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 4, figsize=(24, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = outputs_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(output_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Output {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示新类别伪标签预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(pseudo_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pseudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 3].imshow(label_color_mapped)
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print4(save_dir, outputs, l1h, sal_img):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_images_np = sal_img.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    # outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 2, figsize=(12, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = outputs_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 0].imshow(output_color_mapped)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Output {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示显著性图
        axs[i, 1].imshow(sal_images_np[i])
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'SalImg {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'sal_module.png'))
    print(f'显著性模块可视化图已保存')
    plt.close()


def print5(save_dir, outputs, l1h, sal_img, sal_img2):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_images_np = sal_img.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_images_np2 = sal_img2.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    # outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = outputs_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 0].imshow(output_color_mapped)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Output {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示显著性图
        axs[i, 1].imshow(sal_images_np[i])
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'SalImg {i + 1}')

        # 显示显著性图2
        axs[i, 2].imshow(sal_images_np2[i])
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'SalImg2 {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'sal_module2.png'))
    print(f'显著性模块可视化图2已保存')
    plt.close()


def print6(save_dir, l1h, sal_imgs, fig_name):
    os.makedirs(save_dir, exist_ok=True)
    sal_imgs_np = sal_imgs.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    c = sal_imgs_np[0].shape[0]
    # 设置图像大小
    fig, axs = plt.subplots(24, c, figsize=(c*6, 42), constrained_layout=True)  # 紧凑布局

    # 循环遍历每张图像并显示
    for i in range(24):
        # c = sal_imgs_np[i].shape[0]
        # 显示各类别的概率分布可视化子图
        for j in range(c):
            ax = axs[i, j]
            ax.imshow(sal_imgs_np[i][j])
            class_name = classes.get(j+(21-c), f'Class {j+(21-c)}')  # 使用类名而不是编号
            if j+(21-c) in img_lvl_class_names[i]:
                ax.set_title(class_name, color='red')
            else:
                ax.set_title(class_name, color='black')
            ax.axis('off')

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'sal_logits_{fig_name}.png'))
    plt.close(fig)

    print(f'图像已保存到 {save_dir}')


def print7(save_dir, l1h, sal_img, sal_imgs, cur_step):
    os.makedirs(save_dir, exist_ok=True)
    sal_images_np = sal_img.detach().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_imgs_np = sal_imgs.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 22, figsize=(22*6, 42), constrained_layout=True)  # 紧凑布局

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示各类别的概率分布可视化子图
        for j in range(21):
            ax = axs[i, j]
            ax.imshow(sal_imgs_np[i][j])
            class_name = classes.get(j, f'Class {j}')  # 使用类名而不是编号
            if j in img_lvl_class_names[i]:
                ax.set_title(class_name, color='red')
            else:
                ax.set_title(class_name, color='black')
            ax.axis('off')

        # 显示显著性图
        axs[i, 21].imshow(sal_images_np[i])
        axs[i, 21].axis('off')  # 关闭坐标轴
        axs[i, 21].set_title(f'SalImg {i + 1}')

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'{cur_step}sal_logits.png'))
    plt.close(fig)

    print(f'图像已保存到 {save_dir}')


def print8(save_dir, l1h, labels_np, preds_np, pred_top_classes, recorded_evaluation, iter):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    pred_class_names = [[classes[i] for i in pred_top_class] for pred_top_class in pred_top_classes]
    # 设置图像大小
    fig, axs = plt.subplots(24, 2, figsize=(12, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        record = recorded_evaluation[i]
        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 0].imshow(label_color_mapped)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {record["foreground_iou"]:.2f} {record["background_iou"]:.2f}')
        # axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {entropy_list[0][i]} {entropy_list[1][i]} {entropy_list[2][i]}')


        # 显示预测图像
        pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = preds_np[i] == class_idx
            pred_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(pred_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        # axs[i, 1].set_title(f'Pred {i + 1}: {record["active_classes"]}, {record["foreground_iou"]}, {record["background_iou"]}')
        # axs[i, 1].set_title(f'Pred {i + 1}: {pred_top_classes[i]}')
        axs[i, 1].set_title(f'Pred {i + 1}: {", ".join(pred_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'old_cam_{iter}.png'))
    print(f'旧模型cam可视化图已保存')
    plt.close()


def print9(save_dir, l1h, labels_np, preds_np, sal_img, pred_top_classes, recorded_evaluation, iter):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_imgs_np = sal_img.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    pred_class_names = [[classes[i] for i in pred_top_class] for pred_top_class in pred_top_classes]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        record = recorded_evaluation[i]
        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 0].imshow(label_color_mapped)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {record["foreground_iou"]:.2f} {record["background_iou"]:.2f}')
        # axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {entropy_list[0][i]} {entropy_list[1][i]} {entropy_list[2][i]}')


        # 显示预测图像
        pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = preds_np[i] == class_idx
            pred_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(pred_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        # axs[i, 1].set_title(f'Pred {i + 1}: {record["active_classes"]}, {record["foreground_iou"]}, {record["background_iou"]}')
        # axs[i, 1].set_title(f'Pred {i + 1}: {pred_top_classes[i]}')
        axs[i, 1].set_title(f'Pred {i + 1}: {", ".join(pred_class_names[i])}')

        # 显示显著性图
        axs[i, 2].imshow(sal_imgs_np[i])
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'SalImg {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'old_cam_{iter}.png'))
    print(f'旧模型cam可视化图已保存')
    plt.close()


def print10(save_dir, l1h, labels_np, preds_np, imgs, pred_top_classes, recorded_evaluation, iter):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    pred_class_names = [[classes[i] for i in pred_top_class] for pred_top_class in pred_top_classes]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        record = recorded_evaluation[i]
        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 0].imshow(label_color_mapped)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {record["foreground_iou"]:.2f} {record["background_iou"]:.2f}')
        # axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {entropy_list[0][i]} {entropy_list[1][i]} {entropy_list[2][i]}')


        # 显示预测图像
        pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = preds_np[i] == class_idx
            pred_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(pred_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        # axs[i, 1].set_title(f'Pred {i + 1}: {record["active_classes"]}, {record["foreground_iou"]}, {record["background_iou"]}')
        # axs[i, 1].set_title(f'Pred {i + 1}: {pred_top_classes[i]}')
        axs[i, 1].set_title(f'Pred {i + 1}: {", ".join(pred_class_names[i])}')

        # 显示原图
        image = imgs[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        axs[i, 2].imshow(normalized_image)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Img {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'old_cam_{iter}.png'))
    print(f'旧模型cam可视化图已保存')
    plt.close()


def print11(save_dir, l1h, labels_np, preds_np, sal_img, imgs, pred_top_classes, iter):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    sal_imgs_np = sal_img.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    pred_class_names = [[classes[i] for i in pred_top_class] for pred_top_class in pred_top_classes]
    # 设置图像大小
    fig, axs = plt.subplots(24, 4, figsize=(24, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # if preds_np[i] != False:
        # 显示原图
        image = imgs[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Img {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(
            f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')
        # axs[i, 0].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])} {entropy_list[0][i]} {entropy_list[1][i]} {entropy_list[2][i]}')

        # 显示预测图像
        pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = preds_np[i] == class_idx
            pred_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(pred_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pred {i + 1}: {", ".join(pred_class_names[i])}')

        # 显示显著性图
        axs[i, 3].imshow(sal_imgs_np[i])
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'SalImg {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'old_cam_{iter}.png'))
    print(f'旧模型cam可视化图已保存')
    plt.close()


def print12(save_dir, preds_np):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 设置图像大小
    fig, axs = plt.subplots(24, 1, figsize=(6, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):

        # 显示预测图像
        pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = preds_np[i] == class_idx
            pred_color_mapped[mask] = colors[class_idx]
        axs[i].imshow(pred_color_mapped)
        axs[i].axis('off')  # 关闭坐标轴
        axs[i].set_title(f'Pred {i + 1}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'pred.png'))
    print(f'旧模型pred可视化图已保存')
    plt.close()


def print13(save_dir, pred_np, index):
    # 这里pred和label一样都是bxwxh
    # 确保保存图像的目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 设置图像大小
    fig, axs = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 显示预测图像
    pred_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_idx in range(21):
        mask = pred_np == class_idx
        pred_color_mapped[mask] = colors[class_idx]
    axs.imshow(pred_color_mapped)
    axs.axis('off')  # 关闭坐标轴
    axs.set_title(f'Pred')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'pred{index}.png'))
    print(f'旧模型pred_{index}可视化图已保存')
    plt.close()


def print14(save_dir, l1h, sal_imgs, imgs, labels_np, index):
    os.makedirs(save_dir, exist_ok=True)
    sal_imgs_np = sal_imgs.cpu().detach().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    labels_np = labels_np.cpu().numpy()
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 类别数
    c = sal_imgs_np[0].shape[0]
    # 设置图像大小
    fig, axs = plt.subplots(24, c+2, figsize=((c+2)*6, 42), constrained_layout=True)  # 紧凑布局

    # 循环遍历每张图像并显示
    for i in range(24):
        # if preds_np[i] != False:
        # 显示原图
        image = imgs[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Img {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(
            f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示各类别的概率分布可视化子图
        for j in range(c):
            ax = axs[i, j+2]
            heatmap = sal_imgs_np[i][j]
            min = heatmap.min().item()
            max = heatmap.max().item()
            normalized_heatmap = (heatmap - min) / (max - min)
            ax.imshow(normalized_heatmap)
            class_name = f'{classes[j+(21-c)]}:{round(min, 1)}:{round(max, 1)}'  # 使用类名而不是编号
            if j+(21-c) in img_lvl_class_names[i]:
                ax.set_title(class_name, color='red')
            else:
                ax.set_title(class_name, color='black')
            ax.axis('off')

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'pseudo_{index}.png'))
    plt.close(fig)

    print(f'图像已保存到 {save_dir}')


def print15(save_dir, logits, images, labels, l1h, entropy_list, cur_epoch, cur_step):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    outputs_np = torch.argmax(logits, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):

        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示模型预测输出图像
        output_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = outputs_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(output_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Output {i + 1}:{entropy_list[i]}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(label_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print16(save_dir, target, pseudo_gt_seg_lx, target_old, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在target_old 新模型pseudo_gt_seg_lx 伪标签target

    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_np = torch.argmax(target, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(pseudo_gt_seg_lx, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_old_np = torch.argmax(target_old, dim=1).cpu().numpy()
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 5, figsize=(30, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示拼接后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(output_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pesudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示新模型预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 3].imshow(pseudo_color_mapped)
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'New {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_old_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 4].imshow(output_color_mapped)
        axs[i, 4].axis('off')  # 关闭坐标轴
        axs[i, 4].set_title(f'Old {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print17(save_dir, confidence_scores, target, pseudo_gt_seg_lx, target_old, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在target_old 新模型pseudo_gt_seg_lx 伪标签target

    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    confidence_scores = confidence_scores.cpu().numpy()
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_np = torch.argmax(target, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(pseudo_gt_seg_lx, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_old_np = torch.argmax(target_old, dim=1).cpu().numpy()
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 6, figsize=(36, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示拼接后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(output_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pesudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示伪标签置信度图

        axs[i, 3].imshow(confidence_scores[i])
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'Confidence {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示新模型预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 4].imshow(pseudo_color_mapped)
        axs[i, 4].axis('off')  # 关闭坐标轴
        axs[i, 4].set_title(f'New {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_old_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 5].imshow(output_color_mapped)
        axs[i, 5].axis('off')  # 关闭坐标轴
        axs[i, 5].set_title(f'Old {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print18(save_dir, target_corrected, target, pseudo_gt_seg_lx, target_old, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在target_old 新模型pseudo_gt_seg_lx 伪标签target

    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_c_np = torch.argmax(target_corrected, dim=1).cpu().numpy()
    target_np = torch.argmax(target, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(pseudo_gt_seg_lx, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_old_np = torch.argmax(target_old, dim=1).cpu().numpy()
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 6, figsize=(36, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示修正后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_c_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(output_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Corrected {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示拼接后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 3].imshow(output_color_mapped)
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'Pesudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示新模型预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 4].imshow(pseudo_color_mapped)
        axs[i, 4].axis('off')  # 关闭坐标轴
        axs[i, 4].set_title(f'New {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_old_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 5].imshow(output_color_mapped)
        axs[i, 5].axis('off')  # 关闭坐标轴
        axs[i, 5].set_title(f'Old {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print19(save_dir, low_confidence, target, pseudo_gt_seg_lx, target_old, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在target_old 新模型pseudo_gt_seg_lx 伪标签target

    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    low_confidence_mask = low_confidence.cpu().numpy()
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_np = torch.argmax(target, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(pseudo_gt_seg_lx, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_old_np = torch.argmax(target_old, dim=1).cpu().numpy()
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 6, figsize=(36, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示拼接后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(output_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pesudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示伪标签的低置信度图
        # output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        # for class_idx in range(21):
        #     mask = target_np[i] == class_idx
        #     output_color_mapped[mask] = colors[class_idx]
        # output_color_mapped[low_confidence_mask[i]] = np.array([255, 255, 255])
        score = low_confidence_mask[i]
        score_min = score.min().item()
        score_max = score.max().item()
        axs[i, 3].imshow(score)
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'Score {i + 1}:{round(score_min, 1)}-{round(score_max, 1)}')

        # 显示新模型预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 4].imshow(pseudo_color_mapped)
        axs[i, 4].axis('off')  # 关闭坐标轴
        axs[i, 4].set_title(f'New {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_old_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 5].imshow(output_color_mapped)
        axs[i, 5].axis('off')  # 关闭坐标轴
        axs[i, 5].set_title(f'Old {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print20(save_dir, low_confidence, c_score, s_score, target, pseudo_gt_seg_lx, target_old, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在target_old 新模型pseudo_gt_seg_lx 伪标签target

    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    low_confidence_mask = low_confidence.cpu().numpy()
    c_score = c_score.cpu().numpy()
    s_score = s_score.cpu().numpy()
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_np = torch.argmax(target, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(pseudo_gt_seg_lx, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    target_old_np = torch.argmax(target_old, dim=1).cpu().numpy()
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 9, figsize=(73, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示拼接后的伪标签
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(output_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Pesudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示伪标签的低置信度图
        # output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        # for class_idx in range(21):
        #     mask = target_np[i] == class_idx
        #     output_color_mapped[mask] = colors[class_idx]
        # output_color_mapped[low_confidence_mask[i]] = np.array([255, 255, 255])
        score = low_confidence_mask[i]
        score_min = score.min().item()
        score_max = score.max().item()
        axs[i, 3].imshow(score)
        axs[i, 3].axis('off')  # 关闭坐标轴
        axs[i, 3].set_title(f'Score {i + 1}:{round(score_min, 1)}-{round(score_max, 1)}')

        # 显示置信度得分
        score = c_score[i]
        score_min = score.min().item()
        score_max = score.max().item()
        axs[i, 4].imshow(score)
        axs[i, 4].axis('off')  # 关闭坐标轴
        axs[i, 4].set_title(f'New_conf {i + 1}:{round(score_min, 1)}-{round(score_max, 1)}')

        # 显示置信度得分
        score = c_score[i]*(-1)
        score_min = score.min().item()
        score_max = score.max().item()
        axs[i, 5].imshow(score)
        axs[i, 5].axis('off')  # 关闭坐标轴
        axs[i, 5].set_title(f'Old_conf {i + 1}:{round(score_min, 1)}-{round(score_max, 1)}')



        # 显示相似度得分
        score = s_score[i]
        score_min = score.min().item()
        score_max = score.max().item()
        axs[i, 6].imshow(score)
        axs[i, 6].axis('off')  # 关闭坐标轴
        axs[i, 6].set_title(f'Similar {i + 1}:{round(score_min, 1)}-{round(score_max, 1)}')

        # 显示新模型预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = pseudo_np[i] == class_idx
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 7].imshow(pseudo_color_mapped)
        axs[i, 7].axis('off')  # 关闭坐标轴
        axs[i, 7].set_title(f'New {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示旧模型预测输出图像
        output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = target_old_np[i] == class_idx
            output_color_mapped[mask] = colors[class_idx]
        axs[i, 8].imshow(output_color_mapped)
        axs[i, 8].axis('off')  # 关闭坐标轴
        axs[i, 8].set_title(f'Old {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'comparison_{cur_epoch}_{cur_step}.png'))
    print(f'Epoch{cur_epoch}Step{cur_step}可视化图已保存')
    plt.close()


def print21(save_dir, l1h, weight, imgs, labels_np, index):
    os.makedirs(save_dir, exist_ok=True)
    weight_np = weight.cpu().detach().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    labels_np = labels_np.cpu().numpy()
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [255, 128, 128],  # 浅红色
        [128, 255, 128],  # 浅绿色
        [128, 128, 255],  # 浅蓝色
        [255, 255, 128],  # 浅黄色
        [255, 128, 255],  # 浅紫色
        [128, 255, 255],  # 浅青色
        [255, 64, 64],  # 明亮红色
        [64, 255, 64],  # 明亮绿色
        [64, 64, 255],  # 明亮蓝色
        [255, 192, 128],  # 橙黄色
        [255, 128, 192],  # 粉红紫色
        [128, 192, 255],  # 天蓝色
        [192, 192, 192],  # 浅灰色
        [192, 255, 128],  # 浅黄绿色
        [128, 255, 192],  # 浅水蓝色
    ])

    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 类别数
    c = weight_np[0].shape[0]
    # 设置图像大小
    fig, axs = plt.subplots(24, c+2, figsize=((c+2)*6, 42), constrained_layout=True)  # 紧凑布局

    # 循环遍历每张图像并显示
    for i in range(24):
        # if preds_np[i] != False:
        # 显示原图
        image = imgs[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')  # 关闭坐标轴
        axs[i, 0].set_title(f'Img {i + 1}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(label_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(
            f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示各类别的概率分布可视化子图
        for j in range(c):
            ax = axs[i, j+2]
            ax.imshow(weight_np[i][j])
            class_name = f'{classes[j]}'  # 使用类名而不是编号
            if j in img_lvl_class_names[i]:
                ax.set_title(class_name, color='red')
            else:
                ax.set_title(class_name, color='black')
            ax.axis('off')

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'pseudo_{index}.png'))
    plt.close(fig)

    print(f'图像已保存到 {save_dir}')


def print22(save_dir, new_class_pseudo, images, labels, l1h, cur_epoch, cur_step):
    # 确保保存图像的目录存在
    # save_dir = './logit_npy/10-10-ov/task0_imgs'
    os.makedirs(save_dir, exist_ok=True)
    # 将tensor转换为numpy数组
    labels_np = labels.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    # outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    pseudo_np = torch.argmax(new_class_pseudo, dim=1).cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    image_level_labels_np = l1h.cpu().numpy()  # 先移动到CPU，再转换为numpy数组
    indices_of_img_lvl_lbl = [
        [index + 1 for index, value in enumerate(image_level_label_np) if int(value) == 1] for
        image_level_label_np in image_level_labels_np]

    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # 获取图像级标签的类名作为title
    img_lvl_class_names = [[classes[j] for j in indice_of_img_lvl_lbl] for indice_of_img_lvl_lbl in
                           indices_of_img_lvl_lbl]
    # 设置图像大小
    fig, axs = plt.subplots(24, 3, figsize=(18, 42), constrained_layout=True)  # 紧凑布局
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # 调整子图间距

    # 定义21个颜色，包括一个黑色用于背景类（索引0），其余为其他类别
    colors = np.array([
        [0, 0, 0],  # 黑色用于背景类
        [255, 0, 0],  # 红色
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],  # 深红色
        [0, 128, 0],  # 深绿色
        [0, 0, 128],  # 深蓝色
        [128, 128, 0],  # 橄榄色
        [128, 0, 128],  # 深紫色
        [0, 128, 128],  # 深青色
        [64, 0, 0],  # 更深的红色
        [0, 64, 0],  # 更深的绿色
        [0, 0, 64],  # 更深的蓝色
        [64, 64, 0],  # 更深的黄色
        [64, 0, 64],  # 更深的紫色
        [0, 64, 64],  # 更深的青色
        [192, 192, 192],  # 灰色
        [128, 128, 0],  # 黄绿色
        [0, 128, 128],  # 水蓝色
    ])

    # 循环遍历每张图像并显示
    for i in range(24):
        # 显示原图
        image = images[i]  # 从 images 中获取单张图片
        normalized_image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        normalized_image = normalized_image.permute(1, 2,
                                                    0).cpu().numpy()  # 将通道顺序从 [C, H, W] 改为 [H, W, C] 并转为 numpy 格式

        # 显示归一化后的原图
        axs[i, 0].imshow(normalized_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i + 1}')

        # # 显示旧模型预测输出图像
        # output_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        # for class_idx in range(21):
        #     mask = outputs_np[i] == class_idx
        #     output_color_mapped[mask] = colors[class_idx]
        # axs[i, 1].imshow(output_color_mapped)
        # axs[i, 1].axis('off')  # 关闭坐标轴
        # axs[i, 1].set_title(f'Output {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示新类别伪标签预测输出图像
        pseudo_color_mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        for class_idx in range(6):
            mask = pseudo_np[i] == class_idx
            if class_idx > 0:
                class_idx += 15
            pseudo_color_mapped[mask] = colors[class_idx]
        axs[i, 1].imshow(pseudo_color_mapped)
        axs[i, 1].axis('off')  # 关闭坐标轴
        axs[i, 1].set_title(f'Pseudo {i + 1}: {", ".join(img_lvl_class_names[i])}')

        # 显示标签图像
        label_color_mapped = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_idx in range(21):
            mask = labels_np[i] == class_idx
            label_color_mapped[mask] = colors[class_idx]
        axs[i, 2].imshow(label_color_mapped)
        axs[i, 2].axis('off')  # 关闭坐标轴
        axs[i, 2].set_title(f'Label {i + 1}: {", ".join(img_lvl_class_names[i])}')

    # 保存整个对比图
    plt.savefig(os.path.join(save_dir, f'5set_{cur_step}step.png'))
    print(f'Step{cur_step}伪标签映射阈值设定1可视化图已保存')
    plt.close()