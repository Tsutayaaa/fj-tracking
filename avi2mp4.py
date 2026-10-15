"""Convert AVI video files to MP4 (H.264 + AAC) using ffmpeg.

Usage:
  python avi2mp4.py <input.avi> [output.mp4]    # single file
  python avi2mp4.py --dir /path/to/dir           # batch convert all .avi
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> None:
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        sys.exit("错误: 未找到 ffmpeg。请安装: brew install ffmpeg")


def convert_one(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        sys.exit(f"错误: 文件不存在: {input_path}")

    if input_path.resolve() == output_path.resolve():
        sys.exit("错误: 输入和输出文件名相同。")

    print(f"转换中: {input_path} → {output_path}")

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            str(output_path),
        ],
        check=True,
    )

    print(f"完成: {output_path}")


def batch_convert(directory: Path) -> None:
    files = sorted(directory.glob("*.avi")) + sorted(directory.glob("*.AVI"))
    if not files:
        print(f"在 {directory} 中未找到 .avi 文件。")
        return

    print(f"找到 {len(files)} 个 AVI 文件，开始转换")
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] ", end="")
        convert_one(f, f.with_suffix(".mp4"))
    print(f"全部完成，共转换 {len(files)} 个文件。")


def main() -> None:
    check_ffmpeg()

    parser = argparse.ArgumentParser(description="AVI → MP4 视频转换工具")
    parser.add_argument("input", nargs="?", help="输入 AVI 文件路径")
    parser.add_argument("output", nargs="?", help="输出 MP4 文件路径（可选）")
    parser.add_argument("--dir", "-d", help="批量转换目录下所有 .avi 文件")
    args = parser.parse_args()

    if args.dir:
        batch_convert(Path(args.dir))
    elif args.input:
        output = Path(args.output) if args.output else Path(args.input).with_suffix(".mp4")
        convert_one(Path(args.input), output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
