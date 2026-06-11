# CDF (5,3) Lifting 小波，完全对齐卷积版多级分解逻辑 + 标准对称边界延拓
import numpy as np

def dwt_1d(signal: np.ndarray) -> np.ndarray:
    """一维 5/3 小波正变换，标准 reflect 对称边界延拓"""
    x = signal.astype(np.float64, copy=False)
    n = x.shape[0]
    even = x[0::2].copy()
    odd = x[1::2].copy()
    n_even = len(even)

    # 右向对称延拓 even
    even_pad = np.pad(even, pad_width=(1, 1), mode="reflect")
    next_even = even_pad[1 : 1 + n_even]

    # Predict 步
    detail = odd - 0.5 * (even + next_even)

    # 左向对称延拓 detail
    detail_pad = np.pad(detail, pad_width=(1, 1), mode="reflect")
    prev_detail = detail_pad[0 : n_even]

    # Update 步
    smooth = even + 0.25 * (prev_detail + detail)

    return np.concatenate([smooth, detail])


def idwt_1d(coeff: np.ndarray) -> np.ndarray:
    """一维 5/3 小波逆变换，与正向边界严格匹配"""
    n = coeff.shape[0]
    half = n // 2
    smooth = coeff[:half].astype(np.float64, copy=False)
    detail = coeff[half:].astype(np.float64, copy=False)
    n_smooth = len(smooth)

    # 反 Update
    detail_pad = np.pad(detail, pad_width=(1, 1), mode="reflect")
    prev_detail = detail_pad[0 : n_smooth]
    even = smooth - 0.25 * (prev_detail + detail)

    # 反 Predict
    even_pad = np.pad(even, pad_width=(1, 1), mode="reflect")
    next_even = even_pad[1 : 1 + n_smooth]
    odd = detail + 0.5 * (even + next_even)

    # 交织还原原序列
    x = np.empty(n, dtype=np.float64)
    x[0::2] = even
    x[1::2] = odd
    return x


def dwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """单层二维 5/3 DWT（行→列 顺序不变）"""
    temp = np.empty_like(block, dtype=np.float64)
    for r in range(block.shape[0]):
        temp[r, :] = dwt_1d(block[r, :])

    out = np.empty_like(temp, dtype=np.float64)
    for c in range(temp.shape[1]):
        out[:, c] = dwt_1d(temp[:, c])
    return out


def idwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """单层二维 5/3 IDWT（列→行 顺序不变）"""
    temp = np.empty_like(block, dtype=np.float64)
    for c in range(block.shape[1]):
        temp[:, c] = idwt_1d(block[:, c])

    out = np.empty_like(temp, dtype=np.float64)
    for r in range(temp.shape[0]):
        out[r, :] = idwt_1d(temp[r, :])
    return out


def DWTOneLevel(data, region):
    """
    单层二维DWT，对齐卷积版接口：接收区域 [r0,c0,r1,c1]
    仅对指定子区域做分解，其余区域保留
    """
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].astype(np.float64)
    sub_data = dwt_2d_1l(sub_data)

    out_data = data.astype(np.float64)
    out_data[r0:r1, c0:c1] = sub_data
    return out_data


def dwt(image: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    多级 5/3 DWT
    【核心改动】完全复用你卷积版的区域迭代逻辑，保证子带划分、尺寸和卷积版100%一致
    """
    current_data = image.astype(np.float64)
    h, w = current_data.shape

    r0, c0 = 0, 0
    r1, c1 = h, w

    for _ in range(levels):
        region = [r0, c0, r1, c1]
        current_data = DWTOneLevel(current_data, region)
        # 区域尺寸逐次减半（和卷积版完全一致）
        r1 = r0 + (r1 - r0) // 2
        c1 = c0 + (c1 - c0) // 2

    return current_data


def IDWTOneLevel(data, region):
    """单层二维IDWT，对齐卷积版区域接口"""
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].astype(np.float64)
    sub_data = idwt_2d_1l(sub_data)

    out_data = data.astype(np.float64)
    out_data[r0:r1, c0:c1] = sub_data
    return out_data


def idwt(coeffs: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    多级 5/3 IDWT
    【核心改动】逆变换迭代顺序、区域计算 完全对齐卷积版
    """
    current_data = coeffs.astype(np.float64)
    h, w = current_data.shape

    for level in reversed(range(levels)):
        size_h = h // (2 ** level)
        size_w = w // (2 ** level)
        region = [0, 0, size_h, size_w]
        current_data = IDWTOneLevel(current_data, region)

    return current_data
