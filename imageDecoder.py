"""Executable decoder module required by the final project.

用法：
python imageDecoder.py image.bit 16 path/to/original.png

也可以在 Python 中直接调用：
from imageDecoder import imageDecoder
psnr = imageDecoder("image.bit", 16, "image.png")
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imgcodec.codec import imageDecoder


def main() -> None:
    """命令行入口：读取参数、调用 decoder、打印 PSNR。"""

    parser = argparse.ArgumentParser(description="Decode image.bit and compute PSNR.")
    parser.add_argument("imageBitFileName", help="Path to the compressed bitstream.")
    parser.add_argument("quantizationStepSize", type=float, help="Quantization step size q.")
    parser.add_argument("orgImageFileName", help="Path to the original image.")
    args = parser.parse_args()

    value = imageDecoder(args.imageBitFileName, args.quantizationStepSize, args.orgImageFileName)
    recon_path = Path(args.imageBitFileName).with_suffix(".recon.png")
    print(f"PSNR D(q) = {value:.4f} dB")
    print(f"Reconstructed image: {recon_path}")


if __name__ == "__main__":
    main()
