"""Executable encoder module required by the final project.

用法：
python imageEncoder.py path/to/image.png 16

也可以在 Python 中直接调用：
from imageEncoder import imageEncoder
bitrate = imageEncoder("image.png", 16)
"""

from __future__ import annotations

import argparse

from imgcodec.codec import imageEncoder


def main() -> None:
    """命令行入口：读取参数、调用 encoder、打印 bitrate。"""

    parser = argparse.ArgumentParser(description="Encode a grayscale image into image.bit.")
    parser.add_argument("--orgImageFileName", help="Path to the original image.")
    parser.add_argument("--quantizationStepSize", type=float, help="Quantization step size q.")
    args = parser.parse_args()

    bitrate = imageEncoder(args.orgImageFileName, args.quantizationStepSize)
    print(f"Bitrate R(q) = {bitrate:.6f} bits/pixel")
    print("Output bitstream: image.bit")


if __name__ == "__main__":
    main()
