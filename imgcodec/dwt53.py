# 实现 CDF (5, 3) 小波的 lifting 形式。课程要求使用 5-level subband decomposition using the (5, 3) wavelet，所以这里提供二维多层正变换和反变换，并保持系数布局为常见的子带拼接格式。

import numpy as np


def dwt_1d(signal: np.ndarray) -> np.ndarray:
    """对一维序列做一次 5/3 小波正变换。

    输入序列被分成偶数位置样本 even 和奇数位置样本 odd。
    lifting 的 predict 步生成高频 d，update 步生成低频 s。
    边界位置使用对称延拓的简化形式：缺少的相邻样本复用边界样本。
    """

    # astype(float64) 可以避免整数除法和溢出；后面量化前保留浮点系数。
    x = signal.astype(np.float64, copy=False)

    # 偶数位置是低频预测的基准，奇数位置经过预测后成为高频细节。
    even = x[0::2].copy()
    odd = x[1::2].copy()

    # 对 512、256 等偶数长度，even 和 odd 长度相同。
    # next_even[i] 表示 odd[i] 右侧的偶数样本，最后一个位置复用最后的 even。
    next_even = np.empty_like(even)
    next_even[:-1] = even[1:]
    next_even[-1] = even[-1]

    # Predict: d_i = odd_i - (even_i + even_{i+1}) / 2。
    detail = odd - 0.5 * (even + next_even)

    # prev_detail[i] 表示 even[i] 左侧的高频样本，最左边复用 detail[0]。
    prev_detail = np.empty_like(detail)
    prev_detail[0] = detail[0]
    prev_detail[1:] = detail[:-1]

    # Update: s_i = even_i + (d_{i-1} + d_i) / 4。
    smooth = even + 0.25 * (prev_detail + detail)

    # 输出前半段放低频，后半段放高频，方便二维子带布局。
    return np.concatenate([smooth, detail])


def idwt_1d(coeff: np.ndarray) -> np.ndarray:
    """对一维 5/3 小波系数做一次反变换。

    反变换严格按照 lifting 的相反顺序执行：
    先撤销 update 得到 even，再撤销 predict 得到 odd。
    """

    n = coeff.shape[0]
    half = n // 2

    # 前半段是低频 smooth，后半段是高频 detail。
    smooth = coeff[:half].astype(np.float64, copy=False)
    detail = coeff[half:].astype(np.float64, copy=False)

    # 反 update 需要 detail 的左邻样本。
    prev_detail = np.empty_like(detail)
    prev_detail[0] = detail[0]
    prev_detail[1:] = detail[:-1]

    # even_i = s_i - (d_{i-1} + d_i) / 4。
    even = smooth - 0.25 * (prev_detail + detail)

    # 反 predict 需要 even 的右邻样本。
    next_even = np.empty_like(even)
    next_even[:-1] = even[1:]
    next_even[-1] = even[-1]

    # odd_i = d_i + (even_i + even_{i+1}) / 2。
    odd = detail + 0.5 * (even + next_even)

    # 把 even/odd 重新交织回原始采样顺序。
    x = np.empty(n, dtype=np.float64)
    x[0::2] = even
    x[1::2] = odd
    return x


def dwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """对一个二维块做一级 5/3 小波分解。

    先沿每一行做一维变换，再沿每一列做一维变换。
    输出布局为：
    左上 LL，右上水平高频，左下垂直高频，右下对角高频。
    """

    temp = np.empty_like(block, dtype=np.float64)
    for r in range(block.shape[0]):
        temp[r, :] = dwt_1d(block[r, :])

    out = np.empty_like(temp, dtype=np.float64)
    for c in range(temp.shape[1]):
        out[:, c] = dwt_1d(temp[:, c])
    return out


def idwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """对一个二维块做一级 5/3 小波重构。

    反变换顺序与正变换相反：先按列反变换，再按行反变换。
    """

    temp = np.empty_like(block, dtype=np.float64)
    for c in range(block.shape[1]):
        temp[:, c] = idwt_1d(block[:, c])

    out = np.empty_like(temp, dtype=np.float64)
    for r in range(temp.shape[0]):
        out[r, :] = idwt_1d(temp[r, :])
    return out


def dwt(image: np.ndarray, levels: int = 5) -> np.ndarray:
    """对灰度图像做多级 5/3 DWT 分解。

    每一级只继续分解当前左上角 LL 子带。
    对 512x512 图像做 5 级分解后，最低频 LL 子带大小为 16x16。
    """

    coeffs = image.astype(np.float64, copy=True)
    height, width = coeffs.shape

    for level in range(levels):
        # 当前要分解的区域是上一层的 LL，尺寸每一级减半。
        h = height // (2**level)
        w = width // (2**level)

        # 只替换左上角区域，其他高频子带保持不动。
        coeffs[:h, :w] = dwt_2d_1l(coeffs[:h, :w])

    return coeffs


def idwt(coeffs: np.ndarray, levels: int = 5) -> np.ndarray:
    """对多级 5/3 DWT 系数做反变换，重构灰度图像。"""

    image = coeffs.astype(np.float64, copy=True)
    height, width = image.shape

    # 反变换必须从最小 LL 区域开始，逐级扩大到完整图像。
    for level in range(levels, 0, -1):
        h = height // (2 ** (level - 1))
        w = width // (2 ** (level - 1))
        image[:h, :w] = idwt_2d_1l(image[:h, :w])

    return image
