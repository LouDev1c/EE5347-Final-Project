"""Prediction, raster scan, zero-tree scan, and inverse scan.

扫描模块对应项目要求中的第 4、5、10 步：
- 对最低频 LL 子带做预测；
- LL 子带使用 raster scan；
- 高频子带使用 zero-tree scan，并产生 zero、EZT、non-zero tokens；
- 解码端根据同样的扫描顺序做 inverse scan。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


ZERO_TOKEN = "Z"
EZT_TOKEN = "E"
SIZE_PREFIX = "S"


@dataclass(frozen=True)
class SubbandPosition:
    """表示一个高频子带系数在零树中的位置。"""

    level: int
    orientation: str
    row: int
    col: int


def lowest_subband_size(shape: Tuple[int, int], levels: int) -> Tuple[int, int]:
    """计算 5 级分解后最低频 LL 子带的尺寸。"""

    return shape[0] // (2**levels), shape[1] // (2**levels)


def predict_ll_subband(ll: np.ndarray) -> np.ndarray:
    """对最低频 LL 子带做简单 DPCM 预测。

    预测器只使用已经扫描过的左邻和上邻像素，因此解码端可以按相同顺序恢复。
    残差通常比原始 LL 系数更集中，有利于后续 Huffman 编码。
    """

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
                # 标准二维预测: A + B - C
                A = ll[r, c - 1]
                B = ll[r - 1, c]
                C = ll[r - 1, c - 1]
                pred = A + B - C
            residual[r, c] = ll[r, c] - pred
    return residual


def inverse_predict_ll_subband(residual: np.ndarray) -> np.ndarray:
    """根据 DPCM 残差恢复最低频 LL 子带。"""

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


def value_to_token(value: int) -> Tuple[str, int | None]:
    """把一个整数系数转成扫描 token。

    0 系数使用 Z token；
    非零系数使用 S<size> token，并把实际幅值交给 bitstream 单独写入。
    """

    value = int(value)
    if value == 0:
        return ZERO_TOKEN, None
    size = abs(value).bit_length()
    return f"{SIZE_PREFIX}{size}", value


def token_is_nonzero(token: str) -> bool:
    """判断 token 是否表示一个非零系数的 size。"""

    return token.startswith(SIZE_PREFIX)


def token_to_size(token: str) -> int:
    """从 S<size> token 中取出 size 数值。"""

    if not token_is_nonzero(token):
        raise ValueError(f"Token {token!r} does not contain a non-zero size.")
    return int(token[len(SIZE_PREFIX) :])


def scan_ll_residual(residual: np.ndarray) -> Tuple[List[str], List[int]]:
    """对 LL 残差做光栅扫描，输出 token 序列和非零幅值列表。"""

    tokens: List[str] = []
    amplitudes: List[int] = []

    for value in residual.ravel(order="C"):
        token, amplitude = value_to_token(int(value))
        tokens.append(token)
        if amplitude is not None:
            amplitudes.append(amplitude)

    return tokens, amplitudes


def inverse_scan_ll(tokens: Sequence[str], amplitudes: Iterable[int], ll_shape: Tuple[int, int]) -> np.ndarray:
    """把 LL 的光栅 token 和幅值恢复为残差矩阵。"""

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


def subband_bounds(shape: Tuple[int, int], level: int, orientation: str) -> Tuple[int, int, int, int]:
    """返回某一级某方向高频子带在完整系数矩阵中的边界。

    level=1 是最细层高频子带，level=5 是最粗层高频子带。
    orientation:
    - H: 右上子带，主要表示水平方向变化；
    - V: 左下子带，主要表示垂直方向变化；
    - D: 右下子带，表示对角方向变化。
    """

    height, width = shape
    sub_h = height // (2**level)
    sub_w = width // (2**level)

    if orientation == "H":
        return 0, sub_h, sub_w, 2 * sub_w
    if orientation == "V":
        return sub_h, 2 * sub_h, 0, sub_w
    if orientation == "D":
        return sub_h, 2 * sub_h, sub_w, 2 * sub_w

    raise ValueError(f"Unknown orientation: {orientation}")


def position_to_global(shape: Tuple[int, int], pos: SubbandPosition) -> Tuple[int, int]:
    """把零树中的局部坐标转换为完整系数矩阵坐标。"""

    r0, _r1, c0, _c1 = subband_bounds(shape, pos.level, pos.orientation)
    return r0 + pos.row, c0 + pos.col


def iter_high_frequency_positions(shape: Tuple[int, int], levels: int) -> Iterable[SubbandPosition]:
    """按 zero-tree scan 顺序遍历全部高频子带位置。

    扫描从最粗层到最细层进行；同一级内按 H、V、D 三个方向；
    每个子带内部使用 raster scan。
    """

    for level in range(levels, 0, -1):
        sub_h = shape[0] // (2**level)
        sub_w = shape[1] // (2**level)
        for orientation in ("H", "V", "D"):
            for r in range(sub_h):
                for c in range(sub_w):
                    yield SubbandPosition(level, orientation, r, c)


def descendants(pos: SubbandPosition) -> List[SubbandPosition]:
    """返回某个高频系数在下一细层同方向子带中的直接孩子。"""

    if pos.level <= 1:
        return []

    child_level = pos.level - 1
    base_r = pos.row * 2
    base_c = pos.col * 2

    return [
        SubbandPosition(child_level, pos.orientation, base_r + dr, base_c + dc)
        for dr in range(2)
        for dc in range(2)
    ]


def subtree_positions(pos: SubbandPosition) -> List[SubbandPosition]:
    """返回某个位置的全部后代位置，不包括它自己。"""

    result: List[SubbandPosition] = []
    q = deque(descendants(pos))
    while q:
        child = q.popleft()
        result.append(child)
        q.extend(descendants(child))
    return result


def subtree_all_zero(coeffs: np.ndarray, pos: SubbandPosition) -> bool:
    """判断当前位置的所有后代是否全为 0。"""
    if pos.level <= 1:
        return False
    shape = coeffs.shape
    all_children = subtree_positions(pos)
    for child in all_children:
        r, c = position_to_global(shape, child)
        if coeffs[r, c] != 0:
            return False
    return True


def scan_high_frequency(coeffs: np.ndarray, levels: int) -> Tuple[List[str], List[int]]:
    """对高频子带进行 zero-tree scan。

    如果当前系数为 0 且整棵后代树也全为 0，则输出 EZT token，
    并把后代标记为已覆盖；否则 0 系数输出 Z，非零系数输出 S<size>。
    """

    tokens: List[str] = []
    amplitudes: List[int] = []
    visited: set[SubbandPosition] = set()
    shape = coeffs.shape

    for pos in iter_high_frequency_positions(shape, levels):
        if pos in visited:
            continue

        gr, gc = position_to_global(shape, pos)
        value = int(coeffs[gr, gc])

        if value != 0:
            token, amplitude = value_to_token(value)
            tokens.append(token)
            amplitudes.append(int(amplitude))
            visited.add(pos)
        elif subtree_all_zero(coeffs, pos):
            tokens.append(EZT_TOKEN)
            visited.add(pos)
            visited.update(subtree_positions(pos))
        else:
            tokens.append(ZERO_TOKEN)
            visited.add(pos)

    return tokens, amplitudes


def inverse_scan_high_frequency(
    tokens: Sequence[str],
    amplitudes: Iterable[int],
    shape: Tuple[int, int],
    levels: int,
) -> np.ndarray:
    """根据 zero-tree scan 的 token 和幅值恢复高频量化系数。"""

    coeffs = np.zeros(shape, dtype=np.int32)
    visited: set[SubbandPosition] = set()
    amp_iter = iter(amplitudes)
    token_index = 0

    for pos in iter_high_frequency_positions(shape, levels):
        if pos in visited:
            continue

        if token_index >= len(tokens):
            raise ValueError("Not enough high-frequency tokens for inverse scan.")

        token = tokens[token_index]
        token_index += 1
        gr, gc = position_to_global(shape, pos)

        if token_is_nonzero(token):
            coeffs[gr, gc] = next(amp_iter)
            visited.add(pos)
        elif token == ZERO_TOKEN:
            coeffs[gr, gc] = 0
            visited.add(pos)
        elif token == EZT_TOKEN:
            coeffs[gr, gc] = 0
            visited.add(pos)
            visited.update(subtree_positions(pos))
        else:
            raise ValueError(f"Unknown scan token: {token}")

    if token_index != len(tokens):
        raise ValueError("Extra high-frequency tokens remain after inverse scan.")

    return coeffs
