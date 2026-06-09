from __future__ import annotations
import argparse

from imgcodec.codec import imageEncoder
from imgcodec.network import send_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode image to bitstream, then send to receiver via TCP.")
    # 编码参数（同原 imageEncoder.py）
    parser.add_argument("--orgImageFileName",default="test_images/bird.png",help="Path to the original image.")
    parser.add_argument("--quantizationStepSize", default=0.1, type=float, help="Quantization step size q.")
    # 网络发送参数（同原 encode_send.py）
    parser.add_argument("--host", default="10.27.238.25", help="Receiver host IP.")
    parser.add_argument("--port", type=int, default=5001, help="Receiver port.")

    args = parser.parse_args()
    bit_file = "image.bit"

    # 1. 执行编码
    print("===== 开始编码 =====")
    bitrate = imageEncoder(args.orgImageFileName, args.quantizationStepSize)
    print(f"Bitrate R(q) = {bitrate:.6f} bits/pixel")
    print(f"Encoded bitstream: {bit_file}")

    # 2. 自动发送
    print("\n===== 开始发送 =====")
    send_file(bit_file, args.host, args.port)
    print(f"Successfully sent {bit_file} to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
