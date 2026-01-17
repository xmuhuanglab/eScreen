import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_meme_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    alphabet = lines[2].split()[1]
    # skip the first 10 lines
    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
            current_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_motif["width"] = len(letter_prob_matrix)
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            if current_motif["width"] < 29:
                current_motif["letter_prob_matrix"] = np.concatenate(
                    (
                        current_motif["letter_prob_matrix"],
                        np.zeros((29 - current_motif["width"], 4)),
                    ),
                    axis=0,
                )
    if current_motif is not None:
        # pad to 29, 4
        motifs.append(current_motif)

    motifs_matrix = np.stack([motif["letter_prob_matrix"]
                             for motif in motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    return motifs_matrix, motif_names

def normalize_motif(x):
    mean = x.mean(dim=2, keepdim=True)  # 形状: (batch_size, 1, channels)
    std = x.std(dim=2, keepdim=True)    # 形状: (batch_size, 1, channels)
    return x-mean

def move_zeros_to_end(arr):
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        non_zero_indices = np.where(arr[i] != 0)[0]
        result[i, :len(non_zero_indices)] = arr[i, non_zero_indices]
    
    return result

def load_pwm(pwm,include_reverse_complement=True):
    
    # load pwm
    motifs, motif_names = parse_meme_file(pwm)
    motifs_rev = motifs[:, ::-1, ::-1].copy()
    
    # construct reverse complement
    motifs = torch.tensor(motifs)
    motifs_rev = torch.tensor(motifs_rev)
    motifs = torch.cat([motifs, motifs_rev], dim=0)
    
    # normalize motif
    motifs = normalize_motif(motifs)
    
    return motifs.permute(0,2,1).float(), motif_names

import numpy as np

def load_pwm_from_meme_c(file_path, max_length=29):
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []  # 存储原始矩阵
    complementary_motifs = []  # 存储互补矩阵
    motif_length = [] # 存储motif的长度
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]
    # 跳过前10行
    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
                complementary_motifs.append(current_complementary_motif)
            current_motif = {"name": line.split()[-1], "letter_prob_matrix": []}
            current_complementary_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            complementary_letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            current_complementary_motif["width"] = current_motif["width"]
            motif_length.append(current_motif["width"])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
                complementary_letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_complementary_motif["letter_prob_matrix"] = np.array(complementary_letter_prob_matrix)[::-1,::-1]
            
            current_motif["width"] = len(letter_prob_matrix)
            current_complementary_motif["width"] = current_motif["width"]
            
            current_motif["letter_prob_matrix"] -= 0.25
            current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            
            current_complementary_motif["letter_prob_matrix"] -= 0.25
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]
            
            # 填充到 max_length
            if current_motif["width"] < max_length:
                pad_length = max_length - current_motif["width"]
                current_motif["letter_prob_matrix"] = np.concatenate(
                    ( current_motif["letter_prob_matrix"], np.zeros((pad_length, 4)), ),
                    axis=0,
                )
                current_complementary_motif["letter_prob_matrix"] = np.concatenate(
                    ( current_complementary_motif["letter_prob_matrix"], np.zeros((pad_length, 4)), ),
                    axis=0,
                )
    if current_motif is not None:
        # 填充到 max_length
        motifs.append(current_motif)
        complementary_motifs.append(current_complementary_motif)

    # 将原始矩阵和互补矩阵堆叠成 numpy 数组
    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    
    motifs_matrix=motifs_matrix.transpose(0,2,1)
    complementary_motifs_matrix=complementary_motifs_matrix.transpose(0,2,1)
    
    return motifs_matrix, complementary_motifs_matrix, motif_names, motif_length


import numpy as np

def load_pwm_from_meme_c1(file_path, max_length=29):
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []  # 存储原始矩阵
    complementary_motifs = []  # 存储互补矩阵
    motif_length = []  # 存储 motif 的长度
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]
    lines = lines[9:]  # 跳过前9行

    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
                complementary_motifs.append(current_complementary_motif)
            current_motif = {"name": line.split()[-1], "letter_prob_matrix": []}
            current_complementary_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            complementary_letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            current_complementary_motif["width"] = current_motif["width"]
            motif_length.append(current_motif["width"])

            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
                complementary_letter_prob_matrix.append(row)

            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_complementary_motif["letter_prob_matrix"] = np.array(complementary_letter_prob_matrix)[::-1, ::-1]

            current_motif["width"] = len(letter_prob_matrix)
            current_complementary_motif["width"] = current_motif["width"]

            # 标准化处理
            current_motif["letter_prob_matrix"] -= 0.25
            current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
            current_motif["letter_prob_matrix"] /= current_motif["width"]

            current_complementary_motif["letter_prob_matrix"] -= 0.25
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]

            # 💥 改动开始：中央对齐填充
            if current_motif["width"] < max_length:
                pad_total = max_length - current_motif["width"]
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                current_motif["letter_prob_matrix"] = np.pad(
                    current_motif["letter_prob_matrix"],
                    ((pad_top, pad_bottom), (0, 0)),
                    mode='constant'
                )
                current_complementary_motif["letter_prob_matrix"] = np.pad(
                    current_complementary_motif["letter_prob_matrix"],
                    ((pad_top, pad_bottom), (0, 0)),
                    mode='constant'
                )
            # 💥 改动结束

    if current_motif is not None:
        motifs.append(current_motif)
        complementary_motifs.append(current_complementary_motif)

    # 堆叠 numpy 矩阵
    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]

    motifs_matrix = motifs_matrix.transpose(0, 2, 1)
    complementary_motifs_matrix = complementary_motifs_matrix.transpose(0, 2, 1)

    return motifs_matrix, complementary_motifs_matrix, motif_names, motif_length

class CenteredMaxPool1D(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, x):
        # x shape: (batch, channel, length)
        padded_x = F.pad(x, (self.padding, self.padding), mode='replicate')
        pooled = F.max_pool1d(padded_x, kernel_size=self.kernel_size, stride=self.stride)
        return pooled