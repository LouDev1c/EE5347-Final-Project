"""Huffman coding helpers.

课程要求 zeros、EZT symbols 和 sizes 使用 Huffman coding。
本模块把扫描结果中的 token 编成 Huffman 比特流；非零幅值由 bitstream.py
用符号-幅值形式单独写入。
"""

from __future__ import annotations

from collections import Counter
import heapq
from typing import Dict, Iterable, List, Tuple


def build_huffman_codes(symbols: Iterable[str]) -> Dict[str, str]:
    """根据 token 序列统计频率并生成 Huffman 码表。"""

    frequencies = Counter(symbols)
    if not frequencies:
        return {}

    # heap 中每个元素是 (频率, 排序编号, 树节点)。
    # 排序编号保证频率相同时 Python 不需要比较 dict/list 等不可比较对象。
    heap: List[Tuple[int, int, object]] = []
    for order, (symbol, freq) in enumerate(frequencies.items()):
        heapq.heappush(heap, (freq, order, symbol))

    # 只有一种 token 时，给它分配单比特码 0。
    if len(heap) == 1:
        return {heap[0][2]: "0"}  # type: ignore[index]

    next_order = len(heap)
    while len(heap) > 1:
        freq1, _order1, node1 = heapq.heappop(heap)
        freq2, _order2, node2 = heapq.heappop(heap)
        parent = (node1, node2)
        heapq.heappush(heap, (freq1 + freq2, next_order, parent))
        next_order += 1

    root = heap[0][2]
    codes: Dict[str, str] = {}

    def walk(node: object, prefix: str) -> None:
        """递归遍历 Huffman 树，左分支补 0，右分支补 1。"""

        if isinstance(node, str):
            codes[node] = prefix
            return
        left, right = node  # type: ignore[misc]
        walk(left, prefix + "0")
        walk(right, prefix + "1")

    walk(root, "")
    return codes


def huffman_encode(symbols: Iterable[str], codes: Dict[str, str]) -> str:
    """把 token 序列替换成 Huffman 比特字符串。"""

    return "".join(codes[symbol] for symbol in symbols)


def huffman_decode(bit_string: str, codes: Dict[str, str], symbol_count: int) -> List[str]:
    """根据 Huffman 码表和 token 数量解码 token 序列。"""

    if symbol_count == 0:
        return []

    reverse_codes = {code: symbol for symbol, code in codes.items()}
    decoded: List[str] = []
    buffer = ""

    for bit in bit_string:
        buffer += bit
        if buffer in reverse_codes:
            decoded.append(reverse_codes[buffer])
            buffer = ""
            if len(decoded) == symbol_count:
                break

    if len(decoded) != symbol_count:
        raise ValueError("Huffman bit stream ended before all symbols were decoded.")

    return decoded


def pack_bits(bit_string: str) -> bytes:
    """把 '0'/'1' 字符串打包成 bytes。

    最后一个字节不足 8 位时在右侧补 0；真实长度会写在 header 中，
    解码时会按真实长度截断，所以 padding 不会被当作有效数据。
    """

    if not bit_string:
        return b""

    padding = (-len(bit_string)) % 8
    padded = bit_string + ("0" * padding)
    return bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))


def unpack_bits(data: bytes, bit_length: int) -> str:
    """把 bytes 还原为指定长度的 '0'/'1' 字符串。"""

    if bit_length == 0:
        return ""

    all_bits = "".join(f"{byte:08b}" for byte in data)
    return all_bits[:bit_length]
