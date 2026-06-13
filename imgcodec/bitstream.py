"""Read and write the compressed image.bit file.

位流文件采用一个简单、可调试的二进制容器：
1. 固定 magic 字符串，用来确认文件类型；
2. 4 字节 header 长度；
3. UTF-8 JSON header，保存图像尺寸、量化步长、Huffman 码表等元数据；
4. Huffman token bytes；
5. amplitude bytes。

这样做的好处是：Huffman 编码确实用于 token 流，同时文件又容易检查和扩展。
"""


import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .huffman import build_huffman_codes, huffman_decode, huffman_encode, pack_bits, unpack_bits
from .scan import token_is_nonzero, token_to_size


MAGIC = b"IVCNC1\n"


def encode_amplitudes(tokens: Sequence[str], amplitudes: Sequence[int]) -> str:
    """把非零幅值编码成符号-幅值 bit string。

    每个非零 token S<size> 后面跟：
    - 1 bit 符号位：0 表示正数，1 表示负数；
    - size bits 幅值绝对值。
    """

    bits: List[str] = []
    amp_index = 0

    for token in tokens:
        if not token_is_nonzero(token):
            continue

        if amp_index >= len(amplitudes):
            raise ValueError("Not enough amplitudes for non-zero tokens.")

        value = int(amplitudes[amp_index])
        amp_index += 1
        size = token_to_size(token)
        magnitude = abs(value)

        if magnitude.bit_length() > size:
            raise ValueError("Amplitude does not fit into its recorded size.")

        sign_bit = "1" if value < 0 else "0"
        bits.append(sign_bit)
        bits.append(f"{magnitude:0{size}b}")

    if amp_index != len(amplitudes):
        raise ValueError("Extra amplitudes remain after encoding.")

    return "".join(bits)


def decode_amplitudes(tokens: Sequence[str], bit_string: str) -> List[int]:
    """从符号-幅值 bit string 中恢复非零幅值。"""

    amplitudes: List[int] = []
    offset = 0

    for token in tokens:
        if not token_is_nonzero(token):
            continue

        size = token_to_size(token)
        needed = 1 + size
        if offset + needed > len(bit_string):
            raise ValueError("Amplitude bit stream ended too early.")

        sign_bit = bit_string[offset]
        magnitude_bits = bit_string[offset + 1 : offset + needed]
        offset += needed

        magnitude = int(magnitude_bits, 2)
        value = -magnitude if sign_bit == "1" else magnitude
        amplitudes.append(value)

    if offset != len(bit_string):
        raise ValueError("Extra amplitude bits remain after decoding.")

    return amplitudes


def write_bitstream(
    path: str | Path,
    tokens: Sequence[str],
    amplitudes: Sequence[int],
    image_shape: Tuple[int, int],
    levels: int,
    q_step: float,
    ll_token_count: int,
    source_name: str,
) -> float:
    """把压缩结果写入 image.bit，并返回实际文件 bit rate。"""

    codes = build_huffman_codes(tokens)
    token_bits = huffman_encode(tokens, codes)
    amplitude_bits = encode_amplitudes(tokens, amplitudes)

    token_bytes = pack_bits(token_bits)
    amplitude_bytes = pack_bits(amplitude_bits)

    header: Dict[str, object] = {
        "version": 1,
        "height": int(image_shape[0]),
        "width": int(image_shape[1]),
        "levels": int(levels),
        "q_step": float(q_step),
        "ll_token_count": int(ll_token_count),
        "total_token_count": int(len(tokens)),
        "token_bit_length": int(len(token_bits)),
        "amplitude_bit_length": int(len(amplitude_bits)),
        "token_byte_length": int(len(token_bytes)),
        "amplitude_byte_length": int(len(amplitude_bytes)),
        "huffman_codes": codes,
        "source_name": source_name,
    }

    header_bytes = json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path = Path(path)

    with path.open("wb") as file:
        file.write(MAGIC)
        file.write(len(header_bytes).to_bytes(4, byteorder="big", signed=False))
        file.write(header_bytes)
        file.write(token_bytes)
        file.write(amplitude_bytes)

    # 课程中的 R(q) 通常按 bits per pixel 计算。
    # 这里使用最终 image.bit 文件大小，包含码表和 header 开销，更接近真实传输代价。
    total_bits = path.stat().st_size * 8
    return total_bits / float(image_shape[0] * image_shape[1])


def read_bitstream_header(path: str | Path) -> Dict[str, object]:
    """只读取 image.bit 的 JSON header，不解码 Huffman 载荷。"""

    path = Path(path)
    data = path.read_bytes()

    if not data.startswith(MAGIC):
        raise ValueError("Input file is not an IVCNC image bitstream.")

    offset = len(MAGIC)
    header_length = int.from_bytes(data[offset : offset + 4], byteorder="big", signed=False)
    offset += 4

    return json.loads(data[offset : offset + header_length].decode("utf-8"))


def read_bitstream(path: str | Path) -> Tuple[Dict[str, object], List[str], List[int]]:
    """读取 image.bit，返回 header、token 序列和幅值序列。"""

    path = Path(path)
    data = path.read_bytes()

    if not data.startswith(MAGIC):
        raise ValueError("Input file is not an IVCNC image bitstream.")

    offset = len(MAGIC)
    header_length = int.from_bytes(data[offset : offset + 4], byteorder="big", signed=False)
    offset += 4

    header = json.loads(data[offset : offset + header_length].decode("utf-8"))
    offset += header_length

    token_byte_length = int(header["token_byte_length"])
    amplitude_byte_length = int(header["amplitude_byte_length"])

    token_bytes = data[offset : offset + token_byte_length]
    offset += token_byte_length
    amplitude_bytes = data[offset : offset + amplitude_byte_length]

    token_bits = unpack_bits(token_bytes, int(header["token_bit_length"]))
    amplitude_bits = unpack_bits(amplitude_bytes, int(header["amplitude_bit_length"]))

    tokens = huffman_decode(token_bits, header["huffman_codes"], int(header["total_token_count"]))
    amplitudes = decode_amplitudes(tokens, amplitude_bits)

    return header, tokens, amplitudes
