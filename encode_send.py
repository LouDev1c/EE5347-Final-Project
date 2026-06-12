# 该程序用于压缩图像并自动发送给接收端
import argparse

from imgcodec.codec import imageEncoder
from imgcodec.network import send_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode image to bitstream, then send to receiver via TCP socket.")
    parser.add_argument("--orgImageFileName",default="test_images/seagull.png",help="Path to the original image.")
    parser.add_argument("--quantizationStepSize", default=128, type=float, help="Quantization step size q.")
    parser.add_argument("--host", default="10.27.238.25", help="Receiver host IP.")
    parser.add_argument("--port", type=int, default=5001, help="Receiver port.")

    args = parser.parse_args()
    bit_file = "image.bit"
    # 图像编码
    print("===== 开始编码 =====")
    bitrate = imageEncoder(args.orgImageFileName, args.quantizationStepSize)
    print(f"Bitrate R(q) = {bitrate:.6f} bits/pixel")
    print(f"Encoded bitstream: {bit_file}")
    # 比特流发送
    print("\n===== 开始发送 =====")
    send_file(bit_file, args.host, args.port)
    print(f"Successfully sent {bit_file} to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
