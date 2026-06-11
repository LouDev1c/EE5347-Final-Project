# 该程序用于接收比特流并解码图像

from __future__ import annotations

import argparse
from pathlib import Path

from imgcodec.bitstream import read_bitstream_header
from imgcodec.codec import imageDecoder
from imgcodec.network import receive_file
from imgcodec.dv_utils import RESULTS_DIR, append_rd_row


def receive_decode_and_record(
    host: str,
    port: int,
    result_dir: str | Path = RESULTS_DIR,
) -> None:
    """接收码流、解码重建、计算指标并写入R-D表格（固定原图路径）"""

    # 1. 先接收文件到最终比特流路径（取消临时文件中转）
    # 先读头部获取信息，再命名；这里沿用原逻辑：先存临时再改名（如需彻底优化可再改）
    temp_bit = Path(result_dir) / f"_incoming.bit"
    receive_file(temp_bit, host, port)

    # 2. 解析码流头部参数
    header = read_bitstream_header(temp_bit)
    q_step = float(header["q_step"])
    h, w = int(header["height"]), int(header["width"])
    img_label = Path(header["source_name"]).stem

    # 3. 重命名比特流文件
    bit_file = Path(result_dir) / f"{img_label}_q{q_step:g}.bit"
    if bit_file.exists():
        bit_file.unlink()
    temp_bit.replace(bit_file)

    # 4. 读取原图 + 解码重建图像
    org_file = Path("test_images") / f"{img_label}.png"
    recon_file = Path(result_dir) / f"{img_label}_q{q_step:g}_recon.png"

    # 5. 计算 PSNR、码率，写入CSV
    psnr = imageDecoder(str(bit_file), q_step, str(org_file), recon_path=recon_file)
    bpp = Path(bit_file).stat().st_size * 8 / float(h * w)
    csv_file = append_rd_row(img_label, q_step, bpp, psnr, result_dir)

    # 6. 控制台输出
    print(f"[接收] 码流文件：{bit_file.resolve()}")
    print(f"[接收] 重建图像：{recon_file.resolve()}")
    print(f"码率 R = {bpp:.6f} bpp")
    print(f"PSNR = {psnr:.4f} dB")
    print(f"数据已写入：{csv_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive one compressed image, decode it, and append R-D metrics to CSV.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on.")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on.")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="Output directory for bit files, recon images, and rd_results.csv.",)
    args = parser.parse_args()

    receive_decode_and_record(args.host, args.port, args.out)


if __name__ == "__main__":
    main()
