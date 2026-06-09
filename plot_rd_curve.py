"""Plot R-D curves from trans_results/rd_results.csv.

本脚本只负责读取 CSV 并绘制 R-D 曲线，不负责编码、传输或解码。
双机演示时，接收端通过 receive_decode.py 逐次收图并填充 CSV；
全部收完后，在任意一台机器上运行本脚本生成曲线图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imgcodec.utils import TRANS_RESULTS_DIR, draw_rd_curve, load_rd_csv_points


def main() -> None:
    """命令行入口：读取 CSV 并绘制 R-D 曲线。"""

    parser = argparse.ArgumentParser(description="Plot R-D curves from rd_results.csv.")
    parser.add_argument(
        "--csv",
        default=str(TRANS_RESULTS_DIR / "rd_results.csv"),
        help="Path to the R-D CSV file.",
    )
    parser.add_argument(
        "--out",
        default=str(TRANS_RESULTS_DIR / "rd_curve.png"),
        help="Output path for the R-D curve image.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points = load_rd_csv_points(csv_path)
    draw_rd_curve(points, out_path)

    print(f"Loaded {len(points)} R-D points from {csv_path.resolve()}")
    print(f"Saved R-D curve: {out_path.resolve()}")


if __name__ == "__main__":
    main()
