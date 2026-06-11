#读取 CSV 并绘制 R-D 曲线，不负责编码、传输或解码。双机演示时，接收端通过 receive_decode.py 逐次收图并填充 CSV；全部收完后，在任意一台机器上运行本脚本生成曲线图。

import argparse
from pathlib import Path

from imgcodec.dv_utils import RESULTS_DIR, draw_rd_curve, load_rd_csv_points


def main() -> None:
    csv_path = Path(str(RESULTS_DIR / "rd_results.csv"))
    out_path = Path(str(RESULTS_DIR / "rd_curve.png"))

    points = load_rd_csv_points(csv_path)
    draw_rd_curve(points, out_path)

    print(f"Loaded {len(points)} R-D points from {csv_path.resolve()}")
    print(f"Saved R-D curve: {out_path.resolve()}")


if __name__ == "__main__":
    main()
