# 实现 CDF (5, 3) 小波的 lifting 形式，标准对称边界延拓
import numpy as np

def dwt_1d(signal: np.ndarray) -> np.ndarray:
    """一维 5/3 小波正变换，标准对称延拓"""
    x = signal.astype(np.float64, copy=False)
    n = x.shape[0]
    even = x[0::2].copy()
    odd = x[1::2].copy()
    n_even = len(even)

    # ========== 修复：标准对称延拓，替代简单复制边界 ==========
    # 构造 even 右延拓: [even[0], even, even[-1]] 对称镜像
    even_pad = np.pad(even, pad_width=(1, 1), mode="reflect")
    next_even = even_pad[1 : 1 + n_even]

    # Predict step
    detail = odd - 0.5 * (even + next_even)

    # 构造 detail 左延拓: [detail[0], detail, detail[-1]] 对称镜像
    detail_pad = np.pad(detail, pad_width=(1, 1), mode="reflect")
    prev_detail = detail_pad[0 : n_even]

    # Update step
    smooth = even + 0.25 * (prev_detail + detail)

    return np.concatenate([smooth, detail])


def idwt_1d(coeff: np.ndarray) -> np.ndarray:
    """一维 5/3 小波逆变换，与正向对称延拓严格对应"""
    n = coeff.shape[0]
    half = n // 2
    smooth = coeff[:half].astype(np.float64, copy=False)
    detail = coeff[half:].astype(np.float64, copy=False)
    n_smooth = len(smooth)

    # 反 Update: 对称延拓 detail
    detail_pad = np.pad(detail, pad_width=(1, 1), mode="reflect")
    prev_detail = detail_pad[0 : n_smooth]
    even = smooth - 0.25 * (prev_detail + detail)

    # 反 Predict: 对称延拓 even
    even_pad = np.pad(even, pad_width=(1, 1), mode="reflect")
    next_even = even_pad[1 : 1 + n_smooth]
    odd = detail + 0.5 * (even + next_even)

    # 交织恢复原序列
    x = np.empty(n, dtype=np.float64)
    x[0::2] = even
    x[1::2] = odd
    return x


def dwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """单层二维 5/3 DWT"""
    temp = np.empty_like(block, dtype=np.float64)
    for r in range(block.shape[0]):
        temp[r, :] = dwt_1d(block[r, :])

    out = np.empty_like(temp, dtype=np.float64)
    for c in range(temp.shape[1]):
        out[:, c] = dwt_1d(temp[:, c])
    return out


def idwt_2d_1l(block: np.ndarray) -> np.ndarray:
    """单层二维 5/3 IDWT"""
    temp = np.empty_like(block, dtype=np.float64)
    for c in range(block.shape[1]):
        temp[:, c] = idwt_1d(block[:, c])

    out = np.empty_like(temp, dtype=np.float64)
    for r in range(temp.shape[0]):
        out[r, :] = idwt_1d(temp[r, :])
    return out


def dwt(image: np.ndarray, levels: int = 5) -> np.ndarray:
    """多级 5/3 DWT，仅分解左上角LL"""
    coeffs = image.astype(np.float64, copy=True)
    height, width = coeffs.shape

    for level in range(levels):
        h = height // (2 ** level)
        w = width // (2 ** level)
        coeffs[:h, :w] = dwt_2d_1l(coeffs[:h, :w])
    return coeffs


def idwt(coeffs: np.ndarray, levels: int = 5) -> np.ndarray:
    """多级 5/3 IDWT，从最内层LL逐级恢复"""
    image = coeffs.astype(np.float64, copy=True)
    height, width = image.shape

    for level in range(levels, 0, -1):
        h = height // (2 ** (level - 1))
        w = width // (2 ** (level - 1))
        image[:h, :w] = idwt_2d_1l(image[:h, :w])
    return image
