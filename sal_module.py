import torch
import torch.nn.functional as F
from plt_output import *
import numpy as np
import re
from collections import defaultdict
import math


def sal_2_pseudo(outputs_old, sal_img, l1h):

    save_dir = './logit_npy/15-5-ov/task1_imgs'
    # print4(save_dir, outputs_old, l1h, sal_img)
    # 获取旧类别的最终预测结果 (每像素点属于旧类的最大激活值及对应类别)
    outputs_old_pred = outputs_old.argmax(dim=1)  # [B, H, W]，每像素的预测旧类别

    # 构建辅助监督 mask：显著性图激活为前景且不属于旧类的区域
    sal_foreground_mask = (sal_img == 255)  # 显著性图中前景的掩码
    not_old_class_mask = (outputs_old_pred == 0)  # 旧模型预测为背景类
    aux_supervision_mask = sal_foreground_mask & not_old_class_mask  # 符合条件的像素

    # 将 l1h 变为与 new_class_sal 匹配的形状
    # l1h: [B, 20]
    batch_size = l1h.shape[0]  # 获取 batch size

    # 初始化全 0 的新 l1h_expanded，形状为 [B, 21]
    l1h_with_zeros = torch.zeros((batch_size, 21), device=l1h.device, dtype=l1h.dtype)
    l1h_with_zeros[:, 1:] = l1h  # 将原 l1h 的内容填充到索引 1: 的位置
    # l1h_with_zeros: [B, 21, 1], new_class_sal: [B, H, W]
    l1h_expanded = l1h_with_zeros.unsqueeze(-1).unsqueeze(-1)  # [B, 21, 1, 1]

    # 构造监督标签
    # aux_supervision_mask: [B, H, W] -> [B, 1, H, W] for broadcasting
    aux_supervision_mask_expanded = aux_supervision_mask.unsqueeze(1)  # [B, 1, H, W]
    aux_supervision_mask_expanded = l1h_expanded * aux_supervision_mask_expanded  # [B, 21, H, W]
    # print5(save_dir, outputs_old, l1h, sal_img, aux_supervision_mask)
    # print(f'aux_supervision_mask: {aux_supervision_mask}')
    # print6(save_dir, l1h, aux_supervision_mask)

    return aux_supervision_mask, aux_supervision_mask_expanded


def sal_2_pseudo1(sal_img, l1h):

    save_dir = './logit_npy/15-5-ov/task1_imgs'
    # print4(save_dir, outputs_old, l1h, sal_img)
    # 获取旧类别的最终预测结果 (每像素点属于旧类的最大激活值及对应类别)
    # outputs_old_pred = outputs_old.argmax(dim=1)  # [B, H, W]，每像素的预测旧类别

    # 构建辅助监督 mask：显著性图激活为前景且不属于旧类的区域
    sal_foreground_mask = (sal_img == 255)  # 显著性图中前景的掩码
    # not_old_class_mask = (outputs_old_pred == 0)  # 旧模型预测为背景类
    # aux_supervision_mask = sal_foreground_mask & not_old_class_mask  # 符合条件的像素

    # 将 l1h 变为与 new_class_sal 匹配的形状
    # l1h: [B, 20]
    batch_size = l1h.shape[0]  # 获取 batch size

    # 初始化全 0 的新 l1h_expanded，形状为 [B, 21]
    l1h_with_zeros = torch.zeros((batch_size, 21), device=l1h.device, dtype=l1h.dtype)
    l1h_with_zeros[:, 1:] = l1h  # 将原 l1h 的内容填充到索引 1: 的位置
    # l1h_with_zeros: [B, 21, 1], new_class_sal: [B, H, W]
    l1h_expanded = l1h_with_zeros.unsqueeze(-1).unsqueeze(-1)  # [B, 21, 1, 1]

    # 构造监督标签
    # aux_supervision_mask: [B, H, W] -> [B, 1, H, W] for broadcasting
    aux_supervision_mask_expanded = sal_foreground_mask.unsqueeze(1)  # [B, 1, H, W]
    aux_supervision_mask_expanded = l1h_expanded * aux_supervision_mask_expanded  # [B, 21, H, W]
    # print5(save_dir, outputs_old, l1h, sal_img, aux_supervision_mask)
    # print(f'aux_supervision_mask: {aux_supervision_mask}')
    # print6(save_dir, l1h, aux_supervision_mask)

    return sal_foreground_mask, aux_supervision_mask_expanded


def get_top_classes(prediction):
    # 在新类别图像上测试旧模型时用到
    # 统计旧模型预测输出的图像类别预测前3排名
    batch_top_classes = []  # 存储每个 batch 的结果

    for img in prediction:  # 遍历 batch 中的每张图像
        # 统计每个类别的像素数，忽略背景类 0
        unique, counts = np.unique(img, return_counts=True)
        class_counts = dict(zip(unique, counts))
        if 0 in class_counts:
            del class_counts[0]  # 移除背景类

        # 根据频率排序，选择前 3 个类别
        sorted_classes = sorted(class_counts, key=class_counts.get, reverse=True)[:5]
        batch_top_classes.append(sorted_classes)

    return batch_top_classes


def cal_entropy(logits):
    pred = torch.argmax(logits, dim=1)
    front_mask = (pred == 0)
    background_mask = ~front_mask

    entropy = -torch.sum(logits * torch.log(logits + 1e-6), dim=1)  # 按类别维度求和
    mean_entropy = entropy.mean(dim=(1, 2))  # 求像素点的平均熵
    # 保留 3 位小数
    mean_entropy_rounded = np.around(mean_entropy.cpu().numpy(), decimals = 5)

    front_entropy = entropy * front_mask  # 取前景部分
    mean_front_entropy = front_entropy.mean(dim=(1, 2))  # 求前景像素点的平均熵
    # 保留 3 位小数
    mean_front_entropy_rounded = np.around(mean_front_entropy.cpu().numpy(), decimals = 5)

    back_entropy = entropy * background_mask  # 取背景部分
    mean_back_entropy = back_entropy.mean(dim=(1, 2))  # 求前景像素点的平均熵
    # 保留 3 位小数
    mean_back_entropy_rounded = np.around(mean_back_entropy.cpu().numpy(), decimals = 5)

    entropy_list = [mean_front_entropy_rounded, mean_back_entropy_rounded, mean_entropy_rounded]
    return entropy_list


def cal_confidence(logits):
    # logits: Tensor of shape (B, C, W, H)
    # Apply softmax along the channel dimension (C) to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Get the maximum probability for each pixel (the confidence of the predicted class)
    max_probabilities, _ = torch.max(probabilities, dim=1)  # Shape: (B, W, H)

    # Calculate the mean confidence for each image in the batch
    mean_confidence = torch.mean(max_probabilities, dim=[1, 2])  # Shape: (B,)

    # Return the mean confidence for the batch
    return mean_confidence


def record_similar(file_path, task):
    return True


def cal_similar(file_path):
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

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # 匹配图片记录
    records = re.findall(r"图片 \d+ 的记录：(.+?)(?=--------------------------------------------------|$)", data, re.S)

    # # 初始化统计字典
    # if task == '10-10':
    #     old_classes = set(range(1, 11))  # 旧类
    #     new_classes = set(range(11, 21))  # 新类
    # elif task == '15-5':
    #     old_classes = set(range(1, 16))  # 旧类
    #     new_classes = set(range(16, 21))  # 新类
    # 使用 lambda: defaultdict(int) 定义一个匿名函数，
    # 当访问到外层字典中不存在的键时，defaultdict 会调用这个匿名函数生成一个新的 defaultdict(int)
    # 如果尝试访问 similarity_counter[10]，而 10 还不存在：
    # similarity_counter 会调用 lambda: defaultdict(int)。
    # 创建并返回一个新的 defaultdict(int) int的默认初始化也就是0，同时将其作为 10 的值
    similarity_counter = defaultdict(lambda: defaultdict(int))  # 双向相似性统计

    # 解析每张图片的预测类和激活类
    for record in records:
        pred_match = re.search(r"预测的前3类: \[(.*?)\]", record)
        active_match = re.search(r"新类图像级标签: \[(.*?)\]", record)

        # 如果匹配到内容，解析为数字列表
        # 把预测的前3类: [6, 1]转化为int[6, 1]
        #   激活类: [19]转化为int[19]
        pred_classes = list(map(int, pred_match.group(1).split(','))) if pred_match and pred_match.group(1).strip() else [0]
        active_classes = list(map(int, active_match.group(1).split(','))) if active_match and active_match.group(1).strip() else []

        # 统计类间的共同出现次数
        for pc in pred_classes:
            for ac in active_classes:
                similarity_counter[pc][ac] += 1
                similarity_counter[ac][pc] += 1

    for cls, similar_classes in similarity_counter.items():

        sorted_similarity = sorted(similar_classes.items(), key=lambda x: x[1], reverse=True)
        show_time = 0
        for ss in sorted_similarity:
            show_time += ss[1]
        # print(show_time)
        for ss in sorted_similarity:
            similar_score = similarity_counter[cls][ss[0]]/show_time
            similarity_counter[cls][ss[0]] = similar_score

    return similarity_counter


def getPesudo(prediction, query_similarity, l1h, sal_imgs):
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

    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i+1 for i in true_classes]

        # 统计pred中每个类别的像素数，忽略背景类 0
        unique, counts = np.unique(pred.cpu(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        if 0 in class_counts:
            del class_counts[0]  # 移除背景类

        # print(f'---------------------------{i}--------------------------------')

        # 根据相似性得分判断生成的伪标签质量
        old_num_sum = 0
        old_num_score = 0
        new_cls = true_classes[0]
        new_cls_name = classes[new_cls]
        for old_cls in class_counts:
            cu_similarScore = (query_similarity[old_cls][new_cls] + query_similarity[new_cls][old_cls])/2
            old_num = class_counts[old_cls]
            old_num_sum += old_num
            old_num_score += old_num*cu_similarScore
        remain_num = 512*512 - old_num_sum
        simiScore = (old_num_score+remain_num)/(512*512)

        sal_img = sal_imgs[i].cpu().numpy()
        sal_foreground = (sal_img == True).sum().item()/(512*512)
        pred_foreground = (pred > 0).sum().item()/(512*512)
        foreground_score = pred_foreground/(sal_foreground+ 1e-6)  # 避免除 0

        if (simiScore > 0.7) and (foreground_score > 0.4):
            # print(f'相似性得分{simiScore}')
            for old_cls in class_counts:
                pred[pred == old_cls] = new_cls
                # print(f'{classes[old_cls]}由{new_cls_name}来替换')
        else:
            # print(f'用显著性图来生成伪标签')

            pred[sal_img == True] = true_classes[0]
            # save_dir = './logit_npy/10-10/task0_imgs'
            # plt_output.print13(save_dir, pred, i)
    return prediction


def getPesudo2(prediction, query_similarity, l1h):
    """初始化给赋的负值"""
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    num_classes = 21  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数


    # 初始化 pseudo 张量
    # pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)
    pseudo_label = torch.full(
        (B, new_classes, W, H),
        fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
        device=prediction.device
    )
    pseudo_min = 1e6
    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i + 1 for i in true_classes]

        # 初始化 pseudo 张量
        # pseudo = torch.zeros((new_classes, W, H), device=pred.device)
        pseudo = torch.full(
            (new_classes, W, H),
            fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
            device=pred.device
        )
        # 对未预测类别的输出进行加权组合
        for ci in range(C, num_classes):  # 遍历每个未预测类别
            if ci not in true_classes:  # 如果该类别不在图像级伪标签中，跳过
                continue
            # 初始化 ci 类别的 pseudo 输出
            pseudo_ci = torch.zeros((W, H), device=pred.device)
            # pseudo_ci = torch.full(
            #     (W, H),
            #     fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
            #     device=pred.device
            # )
            # 获取该类别的相似性得分 [C]
            similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数
            for cj in similarity_classes:
                similar_score = (similarity_classes[cj] + query_similarity[cj][ci])/2
                # 对 prediction 中 cj 类别通道加权并累加
                pseudo_ci += similar_score * pred[cj, :, :]
            # 更新一下pseudo_min
            if pseudo_ci.min() < pseudo_min:
                pseudo_min = pseudo_ci.min()
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            pseudo[ci - C, :, :] = pseudo_ci
        pseudo_label[i, :, :, :] = pseudo
    # print(pseudo_label.shape)

    return pseudo_label, pseudo_min


def getPesudo3(prediction, query_similarity, l1h):
    """给每个新类都按权重加权"""
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    num_classes = 21  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数


    # 初始化 pseudo 张量
    pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)

    # 对未预测类别的输出进行加权组合
    for ci in range(C, num_classes):  # 遍历每个未预测类别
        # 初始化 ci 类别的 pseudo 输出
        pseudo_ci = torch.zeros((B, W, H), device=prediction.device)
        # 获取该类别的相似性得分 [C]
        similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数
        for cj in similarity_classes:
            similar_score = (similarity_classes[cj]*0.8 + query_similarity[cj][ci]*0.2)
            # 对 prediction 中 cj 类别通道加权并累加
            pseudo_ci += similar_score * prediction[:, cj, :, :]
        # 将 ci 类别的结果保存到 pseudo 的对应位置
        pseudo_label[:, ci - C, :, :] = pseudo_ci

    return pseudo_label


def getPesudo4(prediction, query_similarity, l1h):
    """给图像级标签的新类全体一块映射到背景类数值范围"""
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    num_classes = 21  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数


    # 初始化 pseudo 张量
    pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)

    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i + 1 for i in true_classes]

        # 初始化 pseudo 张量
        pseudo = torch.zeros((new_classes, W, H), device=pred.device)

        # 存一下图像级标签新类 pseudo 的总范围后面用
        pseudo_min = float('inf')
        pseudo_max = float('-inf')
        # 对未预测类别的输出进行加权组合
        for ci in range(C, num_classes):  # 遍历每个新类别
            # 初始化 ci 类别的 pseudo 输出
            pseudo_ci = torch.zeros((W, H), device=pred.device)
            # 获取该类别的相似性得分 [C]
            similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数
            for cj in similarity_classes:
                similar_score = (similarity_classes[cj] + query_similarity[cj][ci])/2
                # 对 prediction 中 cj 类别通道加权并累加
                pseudo_ci += similar_score * pred[cj, :, :]
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            pseudo[ci - C, :, :] = pseudo_ci

            if ci in true_classes:
                # 更新一下图像级标签新类 pseudo 的总范围
                pseudo_min = min(pseudo_min, pseudo_ci.min().item())
                pseudo_max = max(pseudo_max, pseudo_ci.max().item())

        # 将图像级标签的新类映射到背景类数值范围内
        # 获取 pred[0] 的数值范围
        pred_min = pred[0].min().item()
        pred_max = pred[0].max().item()
        # 遍历 true_classes 将全体 pseudo_ci 映射到 pred[0] 的范围
        for ci in true_classes:
            pseudo_ci = pseudo[ci - C]
            # 使用总范围归一化
            pseudo_ci_rescaled = (pseudo_ci - pseudo_min) / (pseudo_max - pseudo_min)  # 归一化到 [0, 1]
            # 映射到 pred[0] 的范围
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            pseudo[ci - C, :, :] = pseudo_ci_rescaled * (pred_max - pred_min) + pred_min

        pseudo_label[i, :, :, :] = pseudo
    # print(pseudo_label.shape)

    return pseudo_label


def getPesudo5(prediction, query_similarity, l1h):
    """给图像级标签的新类每个都映射到背景类数值范围加激活权重
        max调成全图最大值
    """
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    # voc
    num_classes = 21  # 总类别数
    # coco
    # num_classes = 81  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数

    # 初始化 pseudo 张量
    pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)

    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i + 1 for i in true_classes]

        # 初始化 pseudo 张量
        pseudo = torch.zeros((new_classes, W, H), device=pred.device)
        # 获取 pred[0] 的数值范围
        # pred_min = pred[0].min().item()
        # pred_max = pred[0].max().item()
        # pred_min = pred.max().item()*100
        # pred_max = 101*pred.max().item() - 99*pred[0].min().item()
        # 对未预测类别的输出进行加权组合
        for ci in range(C, num_classes):  # 遍历每个新类别
            # 初始化 ci 类别的 pseudo 输出
            pseudo_ci = torch.zeros((W, H), device=pred.device)
            # 获取该类别的相似性得分 [C]
            similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数,也就是记录在similiar里面与该新类相关的旧类
            for cj in similarity_classes:
                similar_score = similarity_classes[cj]
                # 对 prediction 中 cj 类别通道加权并累加
                pseudo_ci += similar_score * pred[cj, :, :]
            cj_with_max_score = max(similarity_classes, key=lambda x: similarity_classes[x])
            pred_similar_min = pred[cj_with_max_score].min().item()
            pred_similar_max = pred[cj_with_max_score].max().item()
            # print(f'新类{ci}最相似的是{cj_with_max_score}:{similarity_classes[cj_with_max_score]}')
            # 对新类ci的logits范围进行映射
            if ci in true_classes:
                # 获取 pseudo_ci 的数值范围
                pseudo_min = pseudo_ci.min().item()
                pseudo_max = pseudo_ci.max().item()

                # 将 pseudo_ci 映射到 pred[0] 的范围
                pseudo_ci_rescaled = (pseudo_ci - pseudo_min) / (pseudo_max - pseudo_min)  # 归一化到 [0, 1]
                # 比较最相似类和背景类的最小值 取较大的
                # if pred_similar_min > pred_min:
                pseudo_ci = pseudo_ci_rescaled * (pred_similar_max - pred_similar_min) + pred_similar_min  # 映射到 最相似类-max 的范围
                # else:
                # pseudo_ci = pseudo_ci_rescaled * (pred_max - pred_min) + pred_min  # 映射到 pred[0].min-max 的范围
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            pseudo[ci - C, :, :] = pseudo_ci
        pseudo_label[i, :, :, :] = pseudo
    # print(pseudo_label.shape)

    return pseudo_label


def getPesudo6(prediction, query_similarity, l1h):
    """给图像级标签的新类以相似度最高的那个权重为1 剩下的按相似度比值x相似度计算权重"""
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    num_classes = 21  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数

    # 初始化 pseudo 张量
    pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)

    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i + 1 for i in true_classes]

        # 初始化 pseudo 张量
        pseudo = torch.zeros((new_classes, W, H), device=pred.device)
        # 对未预测类别的输出进行加权组合
        for ci in range(C, num_classes):  # 遍历每个新类别
            # 初始化 ci 类别的 pseudo 输出
            pseudo_ci = torch.zeros((W, H), device=pred.device)
            # 获取该类别的相似性得分 [C]
            similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数
            for cj in similarity_classes:
                similar_score = similarity_classes[cj]
                # if similar_score > most_similar_score:
                #     most_similar_score = similar_score
                # 对 prediction 中 cj 类别通道加权并累加
                pseudo_ci += similar_score * pred[cj, :, :]
            # if ci in true_classes:
            #     # 获取 pseudo_ci 的数值范围
            #     pseudo_min = pseudo_ci.min().item()
            #     pseudo_max = pseudo_ci.max().item()
            #
            #     # 将 pseudo_ci 映射到 pred[0] 的范围
            #     pseudo_ci_rescaled = (pseudo_ci - pseudo_min) / (pseudo_max - pseudo_min)  # 归一化到 [0, 1]
            #     pseudo_ci = pseudo_ci_rescaled * (pred_max - pred_min) + pred_min  # 映射到 pred[0] 的范围
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            pseudo[ci - C, :, :] = pseudo_ci / similarity_classes[0]
        pseudo_label[i, :, :, :] = pseudo
    # print(pseudo_label.shape)

    return pseudo_label


def getPesudo7(prediction, query_similarity, l1h):
    """只给图像级标签的新类映射到背景类数值范围加激活权重"""
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

    # 假设输入
    B, C, W, H = prediction.shape  # 当前预测张量的形状
    num_classes = 21  # 总类别数
    new_classes = num_classes - C  # 未预测的类别数

    # 初始化 pseudo 张量
    # pseudo_label = torch.zeros((B, new_classes, W, H), device=prediction.device)
    pseudo_label = torch.full(
        (B, new_classes, W, H),
        fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
        device=prediction.device
    )

    for i, (pred, img_label) in enumerate(zip(prediction, l1h)):
        # 获取图像级标签中激活的类别
        true_classes = torch.nonzero(img_label, as_tuple=False).squeeze(1).tolist()  # 获取非零类别的索引
        true_classes = [i + 1 for i in true_classes]

        # 初始化 pseudo 张量
        # pseudo = torch.zeros((new_classes, W, H), device=pred.device)
        pseudo = torch.full(
            (new_classes, W, H),
            fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
            device=pred.device
        )
        # 获取 pred[0] 的数值范围
        pred_min = pred[0].min().item()
        pred_max = pred[0].max().item()
        # 对未预测类别的输出进行加权组合
        for ci in range(C, num_classes):  # 遍历每个新类别
            # 如果是图像级标签中的类别才进行加权组合
            if ci in true_classes:
                # 初始化 ci 类别的 pseudo 输出
                pseudo_ci = torch.zeros((W, H), device=pred.device)
                # 获取该类别的相似性得分 [C]
                similarity_classes = query_similarity[ci]  # 只取现有类别的相似性分数
                for cj in similarity_classes:
                    similar_score = similarity_classes[cj]
                    # 对 prediction 中 cj 类别通道加权并累加
                    pseudo_ci += similar_score * pred[cj, :, :]
                # if ci in true_classes:
                # 获取 pseudo_ci 的数值范围
                pseudo_min = pseudo_ci.min().item()
                pseudo_max = pseudo_ci.max().item()

                # 将 pseudo_ci 映射到 pred[0] 的范围
                pseudo_ci_rescaled = (pseudo_ci - pseudo_min) / (pseudo_max - pseudo_min)  # 归一化到 [0, 1]
                pseudo_ci = pseudo_ci_rescaled * (pred_max - pred_min) + pred_min  # 映射到 pred[0] 的范
                pseudo[ci - C, :, :] = pseudo_ci
            # else:
            #     # 把pseudo_ci初始化为很小的值不影响最后取argmax
            #     pseudo_ci = torch.zeros((W, H), device=pred.device)
            #     pseudo = torch.full(
            #         (new_classes, W, H),
            #         fill_value=-1e6,  # 你可以修改这个值为其他较大的负数
            #         device=pred.device
            #     )
            # 将 ci 类别的结果保存到 pseudo 的对应位置
            # pseudo[ci - C, :, :] = pseudo_ci
        pseudo_label[i, :, :, :] = pseudo
    # print(pseudo_label.shape)

    return pseudo_label

def normalized(pseudo_label):
    """对pseudo_label上的每个类的热力图自己最大-最小归一化"""
    for i in range(pseudo_label.shape[0]):
        for j in range(pseudo_label[i].shape[0]):
            heatmap = pseudo_label[i][j]
            min = heatmap.min().item()
            max = heatmap.max().item()
            pseudo_label[i][j] = (heatmap - min) / (max - min)
    return pseudo_label


def pseudo_correct(target, target_old, target_new, query_similarity):
    # 旧模型target_old:0-旧类部分 其中0背景类取的较小值 新模型target_new:0+pseudo_gt_seg_lx 伪标签target+背景类
    # target_class = torch.argmax(target, dim=1)  # 最终的分割预测
    target_prob, target_class = target.max(dim=1)
    B, C, W, H = target.shape  # 当前预测张量的形状
    old_prob, old_class = target_old.max(dim=1)
    B, C_old, W, H = target.shape  # 当前预测张量的形状
    new_prob, new_class = target_new.max(dim=1)

    # 使用F.one_hot生成one-hot编码（默认输出维度为[b, c, h, w]）
    new_class_one_hot = F.one_hot(new_class, num_classes=C).int()
    old_class_one_hot = F.one_hot(old_class, num_classes=C_old).int()

    # 找出新旧模型预测冲突的像素点
    mask = (old_class != 0) & (target_class != 0) & (old_class != target_class)
    # 冲突点对应的旧类索引
    # weight = F.one_hot(old_class * mask, num_classes=C_old).int()
    # print(f'shape of weight:{weight.shape}')
    # B, W, H = 2, 32, 32  # batch size 和图像大小
    # num_old_classes = 16
    # num_total_classes = 20

    # 生成 old_class 和 mask
    # old_class = torch.randint(0, num_old_classes, (B, W, H))  # 随机生成类别索引
    # mask = torch.randint(0, 2, (B, W, H))  # 01 掩码

    # One-hot 编码
    one_hot_old = F.one_hot(old_class, num_classes=C_old)  # (B, W, H, 16)
    one_hot_old = one_hot_old.permute(0, 3, 1, 2)  # (B, 16, W, H)

    # 扩展到 20 类
    one_hot_full = torch.zeros((B, C, W, H), dtype=torch.float32)
    one_hot_full[:, :C_old] = one_hot_old  # 只填充前 16 类，后 4 类保持 0

    # 确保 one_hot_full 在与 mask 相同的设备
    one_hot_full = one_hot_full.to(mask.device)
    # 应用 mask，扩展 mask 形状以匹配 (B, 20, W, H)
    weight = one_hot_full * mask.unsqueeze(1)  # 逐通道应用掩码

    # print(masked_one_hot.shape)  # 输出: (B, 20, W, H)

    # mask_old = (old_prob <= new_prob) & mask  # 旧模型预测错误的mask
    # mask_new = (old_prob > new_prob) & mask  # 新模型预测错误的mask
    # wieght = old_class_one_hot[]
    # target_class[mask_old]
    # 将mask扩展为与target相同的形状，方便比较
    # mask_expanded = mask.unsqueeze(1).expand_as(target)
    # # 创建一个与target相同形状的权重矩阵，初始为1
    # weight = torch.ones_like(target, dtype=torch.float32)
    # 对于这些像素，比较prob，保留较大的，较小的设为1

    # weight[mask_expanded] = torch.where(
    #     target*target_class_one_hot > target_old*old_class_one_hot,
    #     old_class_one_hot,
    #     target_class_one_hot
    # )
    # weight = 1 - weight  # 0/1反转

    # foreground_mask = (target_class != 0)  # 前景mask
    # 计算logit分数差距作为置信度
    confidence_scores = torch.abs(old_prob - new_prob)

    # query_similarity[0][0] = 0  # 背景到背景不考虑
    B, C_new, H, W = target_new.shape
    similar_scores = torch.zeros((B, W, H), device=target_new.device)
    # target_weights = torch.zeros((B, W, H), device=target_new.device)
    for i in range(B):
        similar_score = torch.zeros((W, H), device=target_new.device)
        # target_weight = torch.zeros((W, H), device=target_new.device)
        nc_o = oc_o = s = 0
        for n in range(W):
            for m in range(H):
                # tc = int(target_class[i][n][m])
                # if (tc > 0):
                nc_n = int(new_class[i][n][m] + (21 - C_new)) if int(new_class[i][n][m]) != 0 else 0
                oc_n = int(old_class[i][n][m])
                s = query_similarity[nc_n][oc_n]
                # if (nc_n != 0) & (oc_n != 0) & (nc_n != oc_n):  # 新旧类分类不同
                #     target_weight[i][n][m] = 0
                if (nc_n == nc_o) & (oc_n == oc_o):
                    s = s
                else:
                    # 更新
                    s = query_similarity[nc_n][oc_n]
                    nc_o = nc_n
                    oc_o = oc_n
                if s < 0.1:
                    score = 100
                else:
                    score = 100 * math.exp(-100*(s-0.1))
                similar_score[n][m] = score
        similar_scores[i] = similar_score  # 0.1-1

    # bs内部归一化
    # confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
    scores = weight

    # scores = similar_scores

    # low_confidence = (scores < 0.1) & (scores > 0)
    # # 低置信度的像素点且分类为旧类的
    # correct_mask = low_confidence & (target_class == old_class)
    #
    # # 找到需要修正的索引
    # nonzero_indices = correct_mask.nonzero(as_tuple=True)  # 非零位置的索引
    # correct_indices = new_class[nonzero_indices] + (21 - C_new)  # 计算新类索引
    #
    # # 提取需要加的修正值
    # correct_add = confidence_scores[nonzero_indices] * 2  # BxWxH 中的修正值
    #
    # # 仅对需要修正的位置进行更新
    # corrected_target = target.clone()  # 复制 target
    # corrected_target[nonzero_indices[0], correct_indices, nonzero_indices[1], nonzero_indices[2]] += correct_add


    # correct_indices = (new_class+(21-C_new)).unsqueeze(1)
    # correct_add = (correct_mask * confidence_scores).unsqueeze(1)
    # corrected_target = target.clone()  # 复制 target
    # corrected_target.scatter_add_(1, correct_indices, 2 * correct_add)






    # low_confidence = (confidence_scores < 0.3) & (confidence_scores > 0) & intersect_foreground_mask
    # print(f'low_confidence.shape:{low_confidence.shape}')
    # B, C, H, W = target.shape
    #
    # # 确保 low_confidence 和 confidence_scores 的形状正确
    # low_confidence = low_confidence.unsqueeze(1)  # (B, 1, H, W)
    # confidence_scores = confidence_scores.unsqueeze(1)  # (B, 1, H, W)
    #
    # # 创建一个卷积核用于提取每个像素的周围8个像素（3x3卷积核，忽略自身）
    # kernel = torch.ones((C, 1, 3, 3), device=target.device)  # (C, 1, 3, 3)
    # kernel[:, :, 1, 1] = 0  # 忽略自身像素
    # #
    # # 计算每个像素点周围8个像素的预测加权和
    # weighted_sum = F.conv2d(target, kernel, padding=1, groups=C)  # (B, C, H, W)
    #
    # # 计算每个像素点周围8个像素的置信度加权和
    # kernel_confidence = torch.ones((1, 1, 3, 3), device=target.device)  # 用于置信度计算的卷积核 (1, 1, 3, 3)
    # kernel_confidence[:, :, 1, 1] = 0  # 忽略自身像素
    # confidence_sum = F.conv2d(confidence_scores, kernel_confidence, padding=1, groups=1)  # (B, 1, H, W)
    #
    # # 避免除以0，将 confidence_sum 中的0替换为1
    # confidence_sum = torch.where(confidence_sum == 0, torch.ones_like(confidence_sum), confidence_sum)
    #
    # # 加权平均
    # weighted_average = weighted_sum / confidence_sum  # (B, C, H, W)
    #
    # # 只修正低置信度的像素
    # corrected_target = target * ~low_confidence + weighted_average * low_confidence

    return weight, confidence_scores, similar_scores

    # # 逐个归一化
    # min_vals = pseudo_label.view(pseudo_label.shape[0], pseudo_label.shape[1], -1).min(dim=-1)[0].view(-1, 1, 1, 1)
    # max_vals = pseudo_label.view(pseudo_label.shape[0], pseudo_label.shape[1], -1).max(dim=-1)[0].view(-1, 1, 1, 1)
    #
    # # 避免除以零
    # range_vals = max_vals - min_vals
    # range_vals[range_vals == 0] = 1e-6  # 防止除零
    #
    # # 归一化到 [0, 1]
    # normalized_pseudo_label = (pseudo_label - min_vals) / range_vals
    # return normalized_pseudo_label

# file_path = 'record-coco-voc.txt'
# similar = cal_similar(file_path)
# print(similar[19])
# classes = {
#         0: 'background',
#         1: 'aeroplane',
#         2: 'bicycle',
#         3: 'bird',
#         4: 'boat',
#         5: 'bottle',
#         6: 'bus',
#         7: 'car',
#         8: 'cat',
#         9: 'chair',
#         10: 'cow',
#         11: 'diningtable',
#         12: 'dog',
#         13: 'horse',
#         14: 'motorbike',
#         15: 'person',
#         16: 'pottedplant',
#         17: 'sheep',
#         18: 'sofa',
#         19: 'train',
#         20: 'tvmonitor'
#     }
# for key in similar:
# #     if key > 15:
#     simi = similar[key]
#     print(f'class{key}: {[(x, simi[x]) for x in simi]}')