# Image and Video Compression and Network Communication Final Project

本项目实现了课程 Final Project 要求的图像压缩、socket 传输、解码重构和 R-D 曲线评估流程。

## 1. 环境

当前实现使用 Python 3.10，依赖：

```bash
pip install -r requirements.txt
```

本机已验证 `numpy` 和 `Pillow` 可用；项目没有使用 `matplotlib`，R-D 曲线由 Pillow 绘制，便于在离线环境中运行。

## 2. 项目结构

```text
imageEncoder.py          # 编码器可执行模块，提供 imageEncoder(...)
imageDecoder.py          # 解码器可执行模块，提供 imageDecoder(...)
send_image.py            # socket sender，发送 image.bit
receive_image.py         # socket receiver，接收 image.bit
generate_test_images.py  # 生成 3 张 512x512 灰度测试图
demo_rd_curve.py         # 批量测试 q 并生成 R-D 曲线
imgcodec/
  dwt53.py               # 5/3 小波 5-level DWT/IDWT
  scan.py                # LL 预测、raster scan、zero-tree scan、inverse scan
  huffman.py             # Huffman 码表、编码和解码
  bitstream.py           # image.bit 二进制格式读写
  network.py             # TCP socket 文件传输函数
  codec.py               # imageEncoder/imageDecoder 主流程
  utils.py               # 图像读写、PSNR、R-D 曲线绘制
```

## 3. 单张图像编码和解码

生成测试图像：

```bash
python generate_test_images.py
```

编码，其中 `16` 是量化步长 q：

```bash
python imageEncoder.py test_images\gradient.png 16
```

输出：

```text
image.bit
Bitrate R(q) = ... bits/pixel
```

解码并计算 PSNR：

```bash
python imageDecoder.py image.bit 16 test_images\gradient.png
```

输出：

```text
image.recon.png
PSNR D(q) = ... dB
```

也可以在 Python 代码中直接调用老师要求的函数形式：

```python
from imageEncoder import imageEncoder
from imageDecoder import imageDecoder

bitrate = imageEncoder("test_images/gradient.png", 16)
psnr = imageDecoder("image.bit", 16, "test_images/gradient.png")
```

## 4. Socket 网络传输

先在一个终端启动接收端：

```bash
python receive_image.py received_image.bit --host 127.0.0.1 --port 5001
```

再在另一个终端发送编码后的位流：

```bash
python send_image.py image.bit --host 127.0.0.1 --port 5001
```

收到后可直接解码：

```bash
python imageDecoder.py received_image.bit 16 test_images\gradient.png
```

## 5. R-D 曲线实验

对 3 张图像和多个 q 值批量测试：

```bash
python demo_rd_curve.py test_images\gradient.png test_images\shapes.png test_images\texture.png --q 8 16 32 64 --out results
```

输出：

```text
results\rd_results.csv
results\rd_curve.png
results\*_recon.png
```

## 6. 已实现的作业要求对应关系

1. `imageEncoder.py` 读取 512x512 灰度图；非 512x512 输入会缩放到 512x512。
2. `imgcodec/dwt53.py` 实现 5-level (5,3) DWT。
3. `imgcodec/codec.py` 按步长 q 量化 DWT 系数。
4. `imgcodec/scan.py` 对最低频 LL 子带做 DPCM 预测。
5. LL 子带 raster scan，高频子带 zero-tree scan，并产生 `Z`、`E`、`S<size>` token。
6. `imgcodec/huffman.py` 对 zero、EZT symbol、size token 做 Huffman 编码。
7. `imageEncoder(...)` 返回实际 `image.bit` 文件 bit rate。
8. `send_image.py` 和 `receive_image.py` 通过 TCP socket 传输压缩位流。
9. `imageDecoder.py` 使用 Huffman decoder 解码 bit stream。
10. `imgcodec/scan.py` 做 inverse scan。
11. `imgcodec/codec.py` 做 inverse quantization。
12. `imgcodec/dwt53.py` 做 inverse DWT。
13. `imageDecoder(...)` 返回 PSNR。
14. `demo_rd_curve.py` 生成 R-D 曲线。

## 7. 提交源码建议

提交 `LASTNAME-FIRSTNAME-source.rar` 时建议包含源码、README、requirements，不包含 `__pycache__`、临时 bit 文件和重构输出图像。`results` 文件夹可作为报告数据参考，也可以重新运行脚本生成。
