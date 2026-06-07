"""Receive a compressed image bitstream over TCP socket."""

from __future__ import annotations

import argparse

from imgcodec.network import receive_file


def main() -> None:
    """命令行 receiver：监听端口并保存收到的位流。"""

    parser = argparse.ArgumentParser(description="Receive a compressed image bitstream.")
    parser.add_argument("output", nargs="?", default="received_image.bit", help="Output bitstream file.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to listen on.")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on.")
    args = parser.parse_args()

    print(f"Listening on {args.host}:{args.port} ...")
    output = receive_file(args.output, args.host, args.port)
    print(f"Received bitstream: {output}")


if __name__ == "__main__":
    main()
