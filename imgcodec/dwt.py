# 实现 CDF (5, 3) 小波的 lifting 形式。课程要求使用 5-level subband decomposition using the (5, 3) wavelet，所以这里提供二维多层正变换和反变换，并保持系数布局为常见的子带拼接格式。

import numpy as np


def dwt_1d(arr, H_0, H_1) -> np.ndarray:
    M, N = arr.shape
    N_half = N // 2

    # 转换为 numpy 数组
    H_0 = np.array(H_0, dtype=np.float64)
    H_1 = np.array(H_1, dtype=np.float64)

    # 1D DWT 边界延拓
    padded = np.pad(arr, ((0, 0), (2, 2)), mode='reflect')

    # 初始化输出的低频 L 和高频 H 分量
    L = np.zeros((M, N_half), dtype=np.float64)
    H = np.zeros((M, N_half), dtype=np.float64)

    #滤波和下采样
    for k in range(5):
        L += H_0[k] * padded[:, k: 2 * N_half + k: 2]
    for k in range(3):
        H += H_1[k] * padded[:, k + 2: 2 * N_half + k + 2: 2]

    # 水平拼接低频与高频部分
    return np.hstack((L, H))


def dwt_2d_1l(data, region, H_0, H_1):
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].astype(np.float64)

    # 1. 对行进行 1D DWT
    sub_data = dwt_1d(sub_data, H_0, H_1)

    # 2. 对列进行 1D DWT (转置后处理行，再转置回来)
    sub_data = dwt_1d(sub_data.T, H_0, H_1).T

    # 将处理结果写回原区域
    out_data = data.astype(np.float64)
    out_data[r0:r1, c0:c1] = sub_data
    return out_data


def dwt_2d(image, num, H_0, H_1):
    current_data = image.astype(np.float64)
    h, w = current_data.shape

    r0, c0 = 0, 0
    r1, c1 = h, w

    # 逐步对左上角的 LL 子带进行迭代分解
    for level in range(num):
        region = [r0, c0, r1, c1]
        current_data = dwt_2d_1l(current_data, region, H_0, H_1)

        r1 = r0 + (r1 - r0) // 2
        c1 = c0 + (c1 - c0) // 2

    return current_data


def idwt_1d(arr, G_0, G_1):
    M, N = arr.shape
    N_half = N // 2

    G_0 = np.array(G_0, dtype=np.float64)
    G_1 = np.array(G_1, dtype=np.float64)

    # 1. 分离低频 L 和高频 H
    L = arr[:, :N_half]
    H = arr[:, N_half:]

    # 2. 上采样
    L_up = np.zeros((M, N), dtype=np.float64)
    H_up = np.zeros((M, N), dtype=np.float64)
    L_up[:, 0::2] = L
    H_up[:, 1::2] = H

    # 3. 边界对称延拓
    L_up_pad = np.pad(L_up, ((0, 0), (2, 2)), mode='reflect')
    H_up_pad = np.pad(H_up, ((0, 0), (2, 2)), mode='reflect')


    X_L = np.zeros((M, N), dtype=np.float64)
    X_H = np.zeros((M, N), dtype=np.float64)

    # 4. 滤波重构
    # G_0 长度为 3，中心对齐
    for k in range(3):
        X_L += G_0[k] * L_up_pad[:, k + 1: k + 1 + N]

    # G_1 长度为 5，中心对齐
    for k in range(5):
        X_H += G_1[k] * H_up_pad[:, k: k + N]

    return X_L + X_H


def idwt_2d_1l(data, region, G_0, G_1):
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].astype(np.float64)

    # 1. 对列进行 1D IDWT (通过转置矩阵实现)
    sub_data = idwt_1d(sub_data.T, G_0, G_1).T

    # 2. 对行进行 1D IDWT
    sub_data = idwt_1d(sub_data, G_0, G_1)

    # 将重构后的数据写回原区域
    out_data = data.astype(np.float64)
    out_data[r0:r1, c0:c1] = sub_data
    return out_data


def idwt_2d(image, num, G_0, G_1):
    current_data = image.astype(np.float64)
    h, w = current_data.shape

    # 逆变换的迭代顺序从LL5开始
    for level in reversed(range(num)):
        # 计算当前层重构的目标大小
        size_h = h // (2 ** level)
        size_w = w // (2 ** level)

        region = [0, 0, size_h, size_w]
        current_data = idwt_2d_1l(current_data, region, G_0, G_1)

    return current_data
