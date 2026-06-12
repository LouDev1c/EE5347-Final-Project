# 将编码与解码的流程整合为 imageEncoder 和 imageDecoder 两个函数
from pathlib import Path

import numpy as np

from .bitstream import read_bitstream, write_bitstream
from .dwt import dwt_2d, idwt_2d
from .scan import (
    inverse_predict_ll_subband,
    inverse_scan_high_frequency,
    inverse_scan_ll,
    lowest_subband_size,
    predict_ll_subband,
    scan_high_frequency,
    scan_ll_residual,
)
from PIL import Image

DEFAULT_LEVELS = 5
DEFAULT_BITSTREAM = "image.bit"
factor = np.sqrt(2.0)


def imageEncoder(orgImageFileName: str, quantizationStepSize: float) -> float:
    print(f"[ENCODER] Start. Image: {orgImageFileName}, q={quantizationStepSize}")

    q_step = float(quantizationStepSize)
    if q_step <= 0:
        raise ValueError("quantizationStepSize must be positive.")

    # 第 1 步：读取 512x512 灰度图像。
    image = np.asarray(Image.open(orgImageFileName).convert("L"), dtype=np.float64)
    if image is None:
        raise ValueError(f"[ERROR] Cannot read image: {orgImageFileName}")
    if image.shape != (512, 512):
        raise ValueError(f"[ERROR] Image size must be 512x512, got {image.shape}")
    print(f"[ENCODER] Image loaded, shape={image.shape}")

    # 第 2 步：做 5-level (5,3) wavelet subband decomposition。
    H_0 = np.array([-0.125, 0.25, 0.75, 0.25, -0.125], dtype=np.float64) * factor
    H_1 = np.array([-0.5, 1.0, -0.5], dtype=np.float64) * factor
    coeffs = dwt_2d(np.array(image, dtype=np.float64), 5, H_0=H_0, H_1=H_1)

    # 第 3 步：用步长 q 对 DWT 系数量化。
    quantized = np.sign(coeffs) * np.floor(np.abs(coeffs) / q_step)
    quantized = quantized.astype(np.int32)
    print(f"量化后非零系数数量: {np.count_nonzero(quantized)}")

    # 第 4 步：对最低频 LL 子带做预测。
    ll_h, ll_w = lowest_subband_size(quantized.shape, DEFAULT_LEVELS)
    ll = quantized[:ll_h, :ll_w]
    ll_residual = predict_ll_subband(ll)

    # 第 5 步：LL 用 raster scan，高频子带用 zero-tree scan。
    ll_tokens, ll_amplitudes = scan_ll_residual(ll_residual)
    hf_tokens, hf_amplitudes = scan_high_frequency(quantized, DEFAULT_LEVELS)

    # token 流只包含 zeros、EZT symbols 和 sizes；幅值流保存非零系数的 amplitude。
    tokens = ll_tokens + hf_tokens
    amplitudes = ll_amplitudes + hf_amplitudes

    # 第 6、7 步：Huffman 编码并记录 bitrate R(q)。
    bitrate = write_bitstream(
        DEFAULT_BITSTREAM,
        tokens=tokens,
        amplitudes=amplitudes,
        image_shape=quantized.shape,
        levels=DEFAULT_LEVELS,
        q_step=q_step,
        ll_token_count=len(ll_tokens),
        source_name=Path(orgImageFileName).name,
    )

    return float(bitrate)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # 计算 PSNR，单位 dB

    original = original.astype(np.float64, copy=False)
    reconstructed = reconstructed.astype(np.float64, copy=False)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / mse)


def save_grayscale_image(array: np.ndarray, path: str | Path) -> None:
    """把重构数组裁剪到 [0,255] 并保存为 8-bit 灰度图。"""

    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def imageDecoder(
    imageBitFileName: str,
    quantizationStepSize: float,
    orgImageFileName: str,
    recon_path: str | Path | None = None,
) -> float:
    """Decode image.bit, reconstruct the image, and return PSNR.

    参数名保留接口形式：
    PSNR = imageDecoder("image.bit", quantizationStepSize, orgImageFileName)
    """
    print(f"[DECODER] Start. Bitstream: {imageBitFileName}, q={quantizationStepSize}")
    # 读比特流并校验
    with open(imageBitFileName, "rb") as f:
        data = f.read()
    if len(data) == 0:
        raise ValueError(f"[ERROR] Bitstream file is empty: {imageBitFileName}")
    print(f"[DECODER] Bitstream size={len(data)} bytes")

    q_step = float(quantizationStepSize)
    if q_step <= 0:
        raise ValueError("quantizationStepSize must be positive.")

    # 第 9 步：Huffman decoder 解出 token 流，同时恢复 amplitude 流。
    header, tokens, amplitudes = read_bitstream(imageBitFileName)

    height = int(header["height"])
    width = int(header["width"])
    levels = int(header["levels"])
    ll_token_count = int(header["ll_token_count"])

    # 文件 header 中也记录了 q；这里检查传入 q 是否一致，避免误解码。
    stored_q = float(header["q_step"])
    if abs(stored_q - q_step) > 1e-9:
        raise ValueError(f"quantizationStepSize={q_step} does not match bitstream q={stored_q}.")

    ll_h, ll_w = lowest_subband_size((height, width), levels)

    # 根据 token 位置拆分 LL 和高频部分。
    ll_tokens = tokens[:ll_token_count]
    hf_tokens = tokens[ll_token_count:]

    ll_nonzero_count = sum(1 for token in ll_tokens if token.startswith("S"))
    ll_amplitudes = amplitudes[:ll_nonzero_count]
    hf_amplitudes = amplitudes[ll_nonzero_count:]

    # 第 10 步：inverse scan 先恢复量化后的 LL 残差和高频子带。
    ll_residual = inverse_scan_ll(ll_tokens, ll_amplitudes, (ll_h, ll_w))
    ll = inverse_predict_ll_subband(ll_residual)
    quantized = inverse_scan_high_frequency(hf_tokens, hf_amplitudes, (height, width), levels)
    quantized[:ll_h, :ll_w] = ll

    # 第 11 步：inverse quantization。
    coeffs = quantized.astype(np.float64) * q_step + np.sign(quantized) * (q_step / 2.0)
    print(f"反量化后非零系数数量: {np.count_nonzero(coeffs)}")

    # 第 12 步：inverse DWT 重构图像。
    G_0 = np.array([0.5, 1.0, 0.5], dtype=np.float64) / factor
    G_1 = np.array([-0.125, -0.25, 0.75, -0.25, -0.125], dtype=np.float64) / factor
    recon_img = idwt_2d(coeffs, num=5, G_0=G_0, G_1=G_1)
    if recon_img.shape != (512, 512):
        raise ValueError(f"[ERROR] Reconstructed image shape wrong: {recon_img.shape}")
    print(f"[DECODER] Reconstructed image shape={recon_img.shape}")

    # 保存重构图像，方便 demo 和报告查看。
    if recon_path is None:
        recon_output = Path(imageBitFileName).with_suffix(".recon.png")
    else:
        recon_output = Path(recon_path)
    save_grayscale_image(recon_img, recon_output)

    # 第 13 步：计算 PSNR D(q)。
    original = np.asarray(Image.open(orgImageFileName).convert("L"), dtype=np.float64)
    return float(psnr(original, recon_img))
