"""Generate three 512x512 grayscale images for quick R-D testing.

这些图像用于没有标准测试图时快速验证流程：
- gradient.png: 平滑渐变；
- shapes.png: 几何边缘和不同灰度块；
- texture.png: 带纹理的复杂图像。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def save_array(array: np.ndarray, path: Path) -> None:
    """把 numpy 数组保存成灰度 PNG。"""

    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    """生成三张测试图像到 test_images 文件夹。"""

    out_dir = Path("test_images")
    out_dir.mkdir(exist_ok=True)

    size = 512
    y, x = np.mgrid[0:size, 0:size]

    # 第一张：平滑渐变，适合观察低频能量集中的情况。
    gradient = 0.55 * x + 0.35 * y
    gradient = gradient / gradient.max() * 255.0
    save_array(gradient, out_dir / "gradient.png")

    # 第二张：几何图形，包含强边缘和大面积平坦区域。
    shapes = Image.new("L", (size, size), 35)
    draw = ImageDraw.Draw(shapes)
    draw.rectangle((40, 50, 230, 250), fill=210)
    draw.ellipse((260, 60, 470, 270), fill=120)
    draw.polygon([(80, 430), (250, 300), (420, 440)], fill=235)
    draw.line((0, 360, 512, 220), fill=80, width=12)
    shapes.save(out_dir / "shapes.png")

    # 第三张：纹理图，模拟更难压缩的高频内容。
    rng = np.random.default_rng(2026)
    base = 120 + 45 * np.sin(x / 11.0) + 35 * np.cos((x + y) / 17.0)
    noise = rng.normal(0, 22, size=(size, size))
    texture = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), mode="L")
    texture = texture.filter(ImageFilter.GaussianBlur(radius=0.4))
    texture.save(out_dir / "texture.png")

    print(f"Generated test images in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
