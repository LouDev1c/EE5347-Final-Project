"""Run encoder/decoder with multiple q values and draw R-D curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
from typing import List, Tuple

from imgcodec.codec import imageDecoder, imageEncoder
from imgcodec.utils import draw_rd_curve


def run_one_image(image_path: Path, q_values: List[float], result_dir: Path) -> List[Tuple[float, float, str]]:
    """对一张图像测试多个 q，并返回 R-D 点。"""

    points: List[Tuple[float, float, str]] = []
    image_label = image_path.stem

    for q in q_values:
        # encoder 固定输出 image.bit；每次测试后复制一份带名字的 bit 文件。
        bitrate = imageEncoder(str(image_path), q)
        psnr_value = imageDecoder("image.bit", q, str(image_path))

        bit_copy = result_dir / f"{image_label}_q{q:g}.bit"
        recon_copy = result_dir / f"{image_label}_q{q:g}_recon.png"
        shutil.copyfile("image.bit", bit_copy)
        shutil.copyfile("image.recon.png", recon_copy)

        points.append((bitrate, psnr_value, image_label))
        print(f"{image_label:12s} q={q:6g}  R={bitrate:10.6f} bpp  PSNR={psnr_value:8.3f} dB")

    return points


def main() -> None:
    """命令行入口：批量生成 R-D 曲线和 CSV 数据。"""

    parser = argparse.ArgumentParser(description="Evaluate R-D curves for several images.")
    parser.add_argument("--images", default=["test_images/gradient.png", "test_images/shapes.png", "test_images/texture.png"], nargs="+", help="Input image paths.")
    parser.add_argument("--q", nargs="+", type=float, default=[4, 8, 16, 32, 64], help="Quantization step sizes.")
    parser.add_argument("--out", default="results", help="Output directory.")
    args = parser.parse_args()

    result_dir = Path(args.out)
    result_dir.mkdir(exist_ok=True)

    all_points: List[Tuple[float, float, str]] = []
    for name in args.images:
        all_points.extend(run_one_image(Path(name), args.q, result_dir))

    csv_path = result_dir / "rd_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "bitrate_bpp", "psnr_db"])
        for bitrate, psnr_value, label in all_points:
            writer.writerow([label, f"{bitrate:.8f}", f"{psnr_value:.8f}"])

    curve_path = result_dir / "rd_curve.png"
    draw_rd_curve(all_points, curve_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved R-D curve: {curve_path}")


if __name__ == "__main__":
    main()
