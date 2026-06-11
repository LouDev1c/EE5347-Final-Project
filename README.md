# Image and Video Compression and Network Communication Final Project

本项目实现了课程 Final Project 要求的图像压缩、socket 传输、解码重构和 R-D 曲线评估流程。

## 1. 环境

当前实现使用 Python 3.10，依赖：

```bash
pip install -r requirements.txt
```

## 2. 项目结构

```text
encode_send.py              # 编码并发送image.bit
receive_decode.py           # 收图、解码、收集数据生成.csv表格，并写入 results 文件夹
plot_rd_curve.py            # 从 results/rd_results.csv 提取数据并绘制 R-D 曲线
imgcodec/
  dwt53.py                  # 5/3 小波 5-level DWT/IDWT
  scan.py                   # LL 预测、raster scan、zero-tree scan、inverse scan
  huffman.py                # Huffman 码表、编码和解码
  bitstream.py              # image.bit 二进制格式读写
  network.py                # TCP socket 文件传输函数
  codec.py                  # imageEncoder/imageDecoder 主流程
  dv_utils.py               # 作图用到的所有函数
test_images/  
results/                    
```

## 3. 运行流程

接收端运行

```text
receive_decode.py
```

发送端先修改参数`orgImageFileName`、`quantizationStepSize`、`host`，再运行

```text
encode_send.py
```

收集完数据之后运行

```text
plot_rd_curve.py
```
