"""Receive a compressed image bitstream, decode it, and record R-D metrics.

双机演示时，接收端每次运行本脚本接收一张图：
1. 通过 socket 接收 sender 发来的 bitstream；
2. 根据 bitstream header 中的 source_name 和 q 命名保存文件；
3. 解码并保存重构图像；
4. 将 image / q / bitrate_bpp / psnr_db 追加写入 trans_results/rd_results.csv。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imgcodec.bitstream import read_bitstream_header
from imgcodec.codec import imageDecoder
from imgcodec.network import receive_file
from imgcodec.utils import (
    TRANS_RESULTS_DIR,
    append_trans_rd_row,
    bitrate_from_bitstream,
    ensure_trans_results_dir,
)


def resolve_org_image(org_image: str | None, image_label: str) -> Path:
    """确定用于 PSNR 计算的原图路径。"""

    if org_image:
        path = Path(org_image)
        if not path.exists():
            raise FileNotFoundError(f"Original image not found: {path.resolve()}")
        return path

    default_path = Path("test_images") / f"{image_label}.png"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Original image path is required. Pass --org-image or place "
        f"{default_path} on the receiver machine."
    )


def receive_decode_and_record(
    org_image: str | None,
    host: str,
    port: int,
    result_dir: str | Path = TRANS_RESULTS_DIR,
) -> None:
    """接收一张图，保存 bit/recon 文件，并追加 R-D 记录。"""

    directory = ensure_trans_results_dir(result_dir)
    incoming_path = directory / "_incoming.bit"

    receive_file(incoming_path, host, port)

    header = read_bitstream_header(incoming_path)
    q_step = float(header["q_step"])
    height = int(header["height"])
    width = int(header["width"])
    image_label = Path(str(header["source_name"])).stem

    bit_path = directory / f"{image_label}_q{q_step:g}.bit"
    if bit_path.exists():
        bit_path.unlink()
    incoming_path.replace(bit_path)

    org_path = resolve_org_image(org_image, image_label)
    recon_path = directory / f"{image_label}_q{q_step:g}_recon.png"

    psnr_value = imageDecoder(str(bit_path), q_step, str(org_path), recon_path=recon_path)
    bitrate = bitrate_from_bitstream(bit_path, height, width)
    csv_path = append_trans_rd_row(image_label, q_step, bitrate, psnr_value, directory)

    print(f"[RECV] Saved bitstream: {bit_path.resolve()}")
    print(f"[RECV] Saved reconstruction: {recon_path.resolve()}")
    print(f"[RECV] Bitrate R(q) = {bitrate:.6f} bits/pixel")
    print(f"[RECV] PSNR D(q) = {psnr_value:.4f} dB")
    print(f"[RECV] Appended row to {csv_path.resolve()}")


def main() -> None:
    """命令行 receiver：接收一张图并完成解码与 R-D 记录。"""

    parser = argparse.ArgumentParser(
        description="Receive one compressed image, decode it, and append R-D metrics to CSV."
    )
    parser.add_argument(
        "--org-image",
        default=None,
        help="Path to the original image for PSNR. Defaults to test_images/<image>.png if present.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to listen on.")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on.")
    parser.add_argument(
        "--out",
        default=str(TRANS_RESULTS_DIR),
        help="Output directory for bit files, recon images, and rd_results.csv.",
    )
    args = parser.parse_args()

    try:
        receive_decode_and_record(args.org_image, args.host, args.port, args.out)
    except Exception as exc:
        print("\n===== 程序异常退出 =====")
        print(f"错误类型: {type(exc).__name__}")
        print(f"错误信息: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
