import numpy as np
import torch
import config

"""
数据增强
# 此方法通过混合同类样本的不同时间段，生成具有局部特征组合的新数据，可有效提升模型对时间局部模式的识别能力。
1.适用于时间序列分类任务（如EEG、传感器数据）
2.数据量较少时增加样本多样性
3.时间段的局部特征比全局顺序更重要的数据适用
"""

def data_augmentation(data, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 增强后的数据和标签
    aug_data = []
    aug_label = []

    N, C, T = data.shape  # 提取维度(样本数, 通道, 时间采样点)
    seg_size = T // config.num_segs   # 每段时间长度
    aug_data_size = config.batch_size // 4  # 每类生成样本数

    # 这个循环遍历所有类别（假设有4个类别），2a为4，2b为2， HGD为4
    for cls in range(4):
        cls_idx = torch.where(label == cls)
        cls_data = data[cls_idx]
        data_size = cls_data.shape[0]
        if data_size == 0 or data_size == 1:  # 如果没有数据或数据不足，则跳过
            continue
        temp_aug_data = torch.zeros((aug_data_size, C, T), device=device)
        for i in range(aug_data_size):  # 生成增强数据
            rand_idx = torch.randint(0, data_size, (config.num_segs,), device=device)
            for j in range(config.num_segs):
                temp_aug_data[i, :, j * seg_size:(j + 1) * seg_size] = cls_data[rand_idx[j], :,
                                                                               j * seg_size:(j + 1) * seg_size]
        aug_data.append(temp_aug_data)
        aug_label.extend([cls] * aug_data_size)

    if len(aug_data) == 0:  #如果没有增强数据，则返回原始数据
        return data, label

    aug_data = torch.cat(aug_data, dim=0)
    aug_label = torch.tensor(aug_label, device=device)
    aug_shuffle = torch.randperm(len(aug_data), device=device)
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    return aug_data, aug_label


