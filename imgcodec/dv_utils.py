"""Utility functions for image I/O, metrics, and small plotting tasks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

RESULTS_DIR = Path("results")
RD_CSV_HEADERS = ("image", "q", "bitrate_bpp", "psnr_db")

# 在 .csv 表格当中新增一行内容
def append_rd_row(
    image: str,
    q: float,
    bitrate_bpp: float,
    psnr_db: float,
    result_dir: str | Path = RESULTS_DIR,
) -> Path:

    csv_path = Path(result_dir) / f"rd_results.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(RD_CSV_HEADERS)
        writer.writerow([image, f"{q:.8f}", f"{bitrate_bpp:.8f}", f"{psnr_db:.8f}"])

    return csv_path

# 从 rd_results.csv 读取 (bitrate, psnr, image) 点，供 R-D 曲线绘制
def load_rd_csv_points(csv_path: str | Path) -> List[Tuple[float, float, str]]:
    csv_path = Path(csv_path)

    points: List[Tuple[float, float, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            points.append((float(row["bitrate_bpp"]), float(row["psnr_db"]), row["image"]))

    return points

# 绘制 R-D 曲线图
def draw_rd_curve(points: Sequence[Tuple[float, float, str]], output_path: str | Path) -> None:
    width, height = 900, 620
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 50, 90

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    if not points:
        draw.text((30, 30), "No R-D points.", fill="black")
        image.save(output_path)
        return

    rates = [p[0] for p in points]
    qualities = [p[1] for p in points]
    min_r, max_r = min(rates), max(rates)
    min_p, max_p = min(qualities), max(qualities)

    # 给坐标轴留一点余量，避免点压在边框上。
    r_pad = max((max_r - min_r) * 0.08, 0.05)
    p_pad = max((max_p - min_p) * 0.08, 0.5)
    min_r, max_r = max(0.0, min_r - r_pad), max_r + r_pad
    min_p, max_p = min_p - p_pad, max_p + p_pad

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    def map_point(rate: float, quality: float) -> Tuple[int, int]:
        # 把 R-D 数据点映射到画布坐标。
        x = plot_left + int((rate - min_r) / (max_r - min_r) * (plot_right - plot_left))
        y = plot_bottom - int((quality - min_p) / (max_p - min_p) * (plot_bottom - plot_top))
        return x, y

    # 坐标轴和网格。
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(30, 30, 30), width=2)
    for i in range(6):
        x = plot_left + int(i * (plot_right - plot_left) / 5)
        y = plot_bottom - int(i * (plot_bottom - plot_top) / 5)
        draw.line((x, plot_top, x, plot_bottom), fill=(225, 225, 225))
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225))

        rate_label = min_r + i * (max_r - min_r) / 5
        psnr_label = min_p + i * (max_p - min_p) / 5
        draw.text((x - 22, plot_bottom + 12), f"{rate_label:.2f}", fill="black")
        draw.text((20, y - 8), f"{psnr_label:.1f}", fill="black")

    # 按 label 分组画线，同一张图像的不同 q 点连接起来。
    groups: dict[str, List[Tuple[float, float]]] = {}
    for rate, quality, label in points:
        groups.setdefault(label, []).append((rate, quality))

    colors = [(30, 100, 190), (200, 80, 70), (50, 150, 95), (160, 90, 180)]
    for group_index, (label, values) in enumerate(groups.items()):
        color = colors[group_index % len(colors)]
        values = sorted(values)
        xy = [map_point(rate, quality) for rate, quality in values]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=3)
        for x, y in xy:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black")

        legend_y = 18 + group_index * 22
        draw.rectangle((width - 210, legend_y + 3, width - 195, legend_y + 18), fill=color)
        draw.text((width - 188, legend_y), label, fill="black")

    draw.text((width // 2 - 80, 15), "R-D Curve", fill="black")
    draw.text((width // 2 - 95, height - 42), "Bitrate R(q), bits/pixel", fill="black")
    draw.text((14, 15), "PSNR D(q), dB", fill="black")

    image.save(output_path)
