"""Send image.bit to a receiver over TCP socket."""

from __future__ import annotations

import argparse

from imgcodec.network import send_file


def main() -> None:
    """命令行 sender：把压缩位流发送给 receiver。"""

    parser = argparse.ArgumentParser(description="Send a compressed image bitstream.")
    parser.add_argument("file", nargs="?", default="image.bit", help="Bitstream file to send.")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver host.")
    parser.add_argument("--port", type=int, default=5001, help="Receiver port.")
    args = parser.parse_args()

    send_file(args.file, args.host, args.port)
    print(f"Sent {args.file} to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
