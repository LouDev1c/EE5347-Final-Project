# scan 用于数据符号化 (E、Z、S)，便于下一步编码
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
from collections import deque

import numpy as np

# 全局Token定义
ZERO_TOKEN = "Z"
EZT_TOKEN = "E"
SIZE_PREFIX = "S"

# 子带方向定义
ORIENTATIONS = ("HL", "LH", "HH")

# 子带系数基本信息：所在级、所在方向、
@dataclass(frozen=True)
class SubbandPosition:
    level: int
    orientation: str
    row: int
    col: int


def lowest_subband_size(shape: Tuple[int, int], levels: int) -> Tuple[int, int]:
    """计算多级分解后最低频 LL 子带的尺寸。"""
    return shape[0] // (2 ** levels), shape[1] // (2 ** levels)


# 对最低频 LL 子带做 DPCM 预测，输出残差
# 因为低频数值较大的分量多，所以用 A+B-C来推测某个自带系数对应的基准值
def predict_ll_subband(ll: np.ndarray) -> np.ndarray:
    """。"""
    residual = np.zeros_like(ll, dtype=np.int32)
    h, w = ll.shape
    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                pred = 0
            elif r == 0:
                pred = ll[r, c - 1]
            elif c == 0:
                pred = ll[r - 1, c]
            else:
                A = ll[r, c - 1]
                B = ll[r - 1, c]
                C = ll[r - 1, c - 1]
                pred = A + B - C
            residual[r, c] = ll[r, c] - pred
    return residual

# 根据 DPCM 残差恢复最低频 LL 子带
def inverse_predict_ll_subband(residual: np.ndarray) -> np.ndarray:
    ll = np.zeros_like(residual, dtype=np.int32)
    h, w = residual.shape
    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                pred = 0
            elif r == 0:
                pred = ll[r, c - 1]
            elif c == 0:
                pred = ll[r - 1, c]
            else:
                A = ll[r, c - 1]
                B = ll[r - 1, c]
                C = ll[r - 1, c - 1]
                pred = A + B - C
            ll[r, c] = residual[r, c] + pred
    return ll


# Token 编码 (Z、S)
def value_to_token(value: int) -> Tuple[str, int | None]:
    value = int(value)
    if value == 0:
        return ZERO_TOKEN, None
    size = abs(value).bit_length()
    return f"{SIZE_PREFIX}{size}", value

# 判断是否为非零系数Token
def token_is_nonzero(token: str) -> bool:
    return token.startswith(SIZE_PREFIX)

# 从S前缀Token提取长度
def token_to_size(token: str) -> int:
    if not token_is_nonzero(token):
        raise ValueError(f"Token {token!r} does not contain a non-zero size.")
    return int(token[len(SIZE_PREFIX):])


# LL残差 光栅扫描
def scan_ll_residual(residual: np.ndarray) -> Tuple[List[str], List[int]]:
    tokens: List[str] = []
    amplitudes: List[int] = []
    for value in residual.ravel(order="C"):
        token, amplitude = value_to_token(int(value))
        tokens.append(token)
        if amplitude is not None:
            amplitudes.append(amplitude)
    return tokens, amplitudes


def inverse_scan_ll(tokens: Sequence[str], amplitudes: Iterable[int], ll_shape: Tuple[int, int]) -> np.ndarray:
    residual = np.zeros(ll_shape, dtype=np.int32)
    amp_iter = iter(amplitudes)
    for index, token in enumerate(tokens):
        r = index // ll_shape[1]
        c = index % ll_shape[1]
        if token_is_nonzero(token):
            residual[r, c] = next(amp_iter)
        else:
            residual[r, c] = 0
    return residual


# 高频子带映射
def subband_bounds(shape: Tuple[int, int], level: int, orientation: str) -> Tuple[int, int, int, int]:
    """
    返回子带全局坐标 (r0, r1, c0, c1)
    HL: 水平高频 右上
    LH: 垂直高频 左下
    HH: 对角高频 右下
    level: 层级，level=1 最细层，level=levels 最粗层
    """
    if level < 1:
        raise ValueError("level 必须大于等于 1")
    height, width = shape
    sub_h = height // (2 ** level)
    sub_w = width // (2 ** level)

    if orientation == "HL":
        return 0, sub_h, sub_w, 2 * sub_w
    elif orientation == "LH":
        return sub_h, 2 * sub_h, 0, sub_w
    elif orientation == "HH":
        return sub_h, 2 * sub_h, sub_w, 2 * sub_w
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

# 子带局部坐标推理全局图像坐标
def position_to_global(shape: Tuple[int, int], pos: SubbandPosition) -> Tuple[int, int]:
    r0, _, c0, _ = subband_bounds(shape, pos.level, pos.orientation)
    return r0 + pos.row, c0 + pos.col


# 高频位置遍历（对齐标准扫描顺序）
def iter_high_frequency_positions(shape: Tuple[int, int], levels: int) -> Iterable[SubbandPosition]:
    """
    零树标准遍历顺序：
    1. 从最粗层(levels) → 最细层(1)
    2. 每层顺序：HL → LH → HH
    3. 子带内部行优先光栅扫描
    """
    for level in range(levels, 0, -1):
        sub_h = shape[0] // (2 ** level)
        sub_w = shape[1] // (2 ** level)
        for orient in ORIENTATIONS:
            for r in range(sub_h):
                for c in range(sub_w):
                    yield SubbandPosition(level, orient, r, c)


# 获取当前节点的直接子节点
def descendants(pos: SubbandPosition) -> List[SubbandPosition]:
    # 最细层级无后代
    if pos.level <= 1:
        return []
    child_level = pos.level - 1
    base_r = pos.row * 2
    base_c = pos.col * 2
    return [
        SubbandPosition(child_level, pos.orientation, base_r + dr, base_c + dc)
        for dr in (0, 1)
        for dc in (0, 1)
    ]

# 广度优先遍历整棵子树，整理这一个节点及其所有子孙节点的列表
def subtree_positions(pos: SubbandPosition) -> List[SubbandPosition]:
    result: List[SubbandPosition] = []
    q = deque(descendants(pos))
    while q:
        child = q.popleft()
        result.append(child)
        q.extend(descendants(child))
    return result

# 判断当前节点和整棵子树是否全部为 0
def subtree_all_zero(coeffs: np.ndarray, pos: SubbandPosition) -> bool:
    # 最细层无后代，不能作为零树根
    if pos.level <= 1:
        return False

    shape = coeffs.shape
    # 先校验自身
    gr, gc = position_to_global(shape, pos)
    if coeffs[gr, gc] != 0:
        return False

    # 再校验所有后代
    for child in subtree_positions(pos):
        cr, cc = position_to_global(shape, child)
        if coeffs[cr, cc] != 0:
            return False
    return True


# 高频零树扫描 / 逆扫描（
def scan_high_frequency(coeffs: np.ndarray, levels: int) -> Tuple[List[str], List[int]]:
    """
    - 非零值 → S + 幅值
    - 零值 + 整树全零 → E (EZT 零树根)，子树全部标记跳过
    - 零值 + 子树存在非零 → Z (普通零)
    """
    tokens: List[str] = []
    amplitudes: List[int] = []
    visited: set[SubbandPosition] = set()
    shape = coeffs.shape

    for pos in iter_high_frequency_positions(shape, levels):
        if pos in visited:
            continue

        gr, gc = position_to_global(shape, pos)
        val = int(coeffs[gr, gc])

        if val != 0:
            # 非零系数
            token, amp = value_to_token(val)
            tokens.append(token)
            amplitudes.append(amp)
            visited.add(pos)
        elif subtree_all_zero(coeffs, pos):
            # 零树根 EZT：自身+所有后代全部跳过
            tokens.append(EZT_TOKEN)
            visited.add(pos)
            visited.update(subtree_positions(pos))
        else:
            # 普通零 Z：仅跳过自身
            tokens.append(ZERO_TOKEN)
            visited.add(pos)

    return tokens, amplitudes


def inverse_scan_high_frequency(
    tokens: Sequence[str],
    amplitudes: Iterable[int],
    shape: Tuple[int, int],
    levels: int,
) -> np.ndarray:
    coeffs = np.zeros(shape, dtype=np.int32)
    visited: set[SubbandPosition] = set()
    amp_iter = iter(amplitudes)
    token_idx = 0
    total_token = len(tokens)

    for pos in iter_high_frequency_positions(shape, levels):
        if pos in visited:
            continue
        if token_idx >= total_token:
            raise ValueError("高频Token数量不足，解码失败")

        curr_token = tokens[token_idx]
        token_idx += 1
        gr, gc = position_to_global(shape, pos)

        if token_is_nonzero(curr_token):
            # 恢复非零系数
            coeffs[gr, gc] = next(amp_iter)
            visited.add(pos)
        elif curr_token == ZERO_TOKEN:
            # 普通零
            coeffs[gr, gc] = 0
            visited.add(pos)
        elif curr_token == EZT_TOKEN:
            # 零树根：自身+整棵子树置0并标记跳过
            coeffs[gr, gc] = 0
            visited.add(pos)
            visited.update(subtree_positions(pos))
        else:
            raise ValueError(f"未知Token: {curr_token}")

    if token_idx != total_token:
        raise ValueError("高频Token有冗余，解码失败")

    return coeffs
