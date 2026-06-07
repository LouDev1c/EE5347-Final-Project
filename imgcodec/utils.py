"""Utility functions for image I/O, metrics, and small plotting tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_SIZE = (512, 512)


def read_grayscale_image(path: str | Path, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """读取图像，转成 512x512 灰度 numpy 数组。

    老师要求输入为 512x512 grayscale image。
    为了演示时更稳，如果输入不是 512x512，本函数会自动缩放到目标尺寸。
    """

    image = Image.open(path).convert("L")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64)


def save_grayscale_image(array: np.ndarray, path: str | Path) -> None:
    """把重构数组裁剪到 [0,255] 并保存为 8-bit 灰度图。"""

    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """计算 PSNR，单位 dB。"""

    original = original.astype(np.float64, copy=False)
    reconstructed = reconstructed.astype(np.float64, copy=False)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / mse)


def draw_rd_curve(points: Sequence[Tuple[float, float, str]], output_path: str | Path) -> None:
    """使用 Pillow 绘制简单 R-D 曲线图。

    这里不用 matplotlib，是为了让项目在当前已有环境下直接运行。
    points 中每项为 (bitrate, psnr, label)。
    """

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
        """把 R-D 数据点映射到画布坐标。"""

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
