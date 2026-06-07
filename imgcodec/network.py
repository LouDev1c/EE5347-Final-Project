"""Socket based bitstream transmission helpers.

网络传输部分对应项目要求第 8 步：
把 encoder 产生的 compressed image data 通过 socket 发送给 receiver。
协议非常简单：
1. sender 先发送 8 字节无符号整数，表示文件长度；
2. sender 再发送 image.bit 的原始 bytes；
3. receiver 按长度接收并写入本地文件。
"""

from __future__ import annotations

from pathlib import Path
import socket
import hashlib


CHUNK_SIZE = 64 * 1024


def send_file(file_path: str | Path, host: str = "127.0.0.1", port: int = 5001) -> None:
    """作为客户端连接 receiver，并发送压缩位流文件。"""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    data = path.read_bytes()

    md5 = hashlib.md5(data).hexdigest()
    print(md5)

    file_size = len(data)
    file_hash = hashlib.md5(data).hexdigest()
    print(f"[SEND] File: {path}, size: {file_size} bytes, MD5: {file_hash}")
    print(f"[SEND] Connecting to {host}:{port}...")

    with socket.create_connection((host, port), timeout=10.0) as sock:
        # 8 字节长度头让接收端知道应该读取多少数据。
        sock.sendall(len(data).to_bytes(8, byteorder="big", signed=False))
        sock.sendall(data)
    print(f"[SEND] Done.")


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    """从 socket 中精确读取 length 字节。"""

    chunks = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(min(CHUNK_SIZE, remaining))
        if not chunk:
            raise ConnectionError("Socket closed before all data was received.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_file(output_path: str | Path, host: str = "127.0.0.1", port: int = 5001) -> Path:
    """作为服务器监听一个连接，并把收到的位流写入 output_path。"""

    output = Path(output_path)
    print(f"[RECV] Listening on {host}:{port} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)

        conn, _addr = server.accept()
        print(f"[RECV] Connected to {_addr}")
        with conn:
            length = int.from_bytes(_recv_exact(conn, 8), byteorder="big", signed=False)
            print(f"[RECV] Expecting {length} bytes...")
            data = _recv_exact(conn, length)

    md5 = hashlib.md5(data).hexdigest()
    print(md5)

    recv_size = len(data)
    recv_hash = hashlib.md5(data).hexdigest()
    print(f"[RECV] Received {recv_size} bytes, MD5: {recv_hash}")
    if recv_size != length:
        raise ValueError(f"[ERROR] Received size mismatch: got {recv_size}, expected {length}")

    output.write_bytes(data)
    print(f"[RECV] Saved to {output.resolve()}")
    return output
