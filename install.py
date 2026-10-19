#!/usr/bin/env python
"""创建隔离的运行环境并安装依赖；不需要在启动环境安装科学计算库。"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="安装细菌追踪管线的 conda 环境")
    parser.add_argument("mode", nargs="?", choices=("both", "omni", "misic"), default="both")
    parser.add_argument("--cuda", action="store_true", help="为 Omnipose 安装 CUDA 12.1 版 PyTorch（Linux/Windows）")
    args = parser.parse_args()
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda or not Path(conda).is_file():
        parser.error("请先安装 Miniconda，并在能够执行 conda 的终端中运行此脚本。")
    if args.cuda and platform.system() not in ("Linux", "Windows"):
        parser.error("--cuda 只适用于带 NVIDIA GPU 的 Linux/Windows。其他平台请使用默认 CPU 安装。")
    omni_env = os.environ.get("OMNI_ENV", "omnipose")
    misic_env = os.environ.get("MISIC_ENV", "misic")
    if omni_env == misic_env or {omni_env, misic_env} & {"base", "root"}:
        parser.error("两个后端必须使用不同环境，且不能使用 base/root。")

    def run(command):
        subprocess.run(command, cwd=str(BASE), check=True)

    def prepare(name, specification):
        result = subprocess.run([conda, "env", "list", "--json"], capture_output=True,
                                text=True, check=True)
        existing = json.loads(result.stdout)["envs"]
        if not any(Path(path).name == name for path in existing):
            run([conda, "env", "create", "-n", name, "-f", str(BASE / specification)])
        command = [conda, "run", "--no-capture-output", "-n", name, "python"]
        run(command + ["-c", "import sys; assert sys.version_info[:2] == (3, 10), '请使用 Python 3.10 的独立环境'"])
        return command + ["-m", "pip", "install"]

    # MiSiC 的清理、追踪和动画也使用此环境，无需为其安装 Omnipose 模型依赖。
    omni_pip = prepare(omni_env, "environment.yml")
    run(omni_pip + ["-r", "requirements.txt"])
    if args.mode in ("both", "omni"):
        torch_args = ["torch==2.5.1", "torchvision==0.20.1"]
        if platform.system() in ("Linux", "Windows"):
            channel = "cu121" if args.cuda else "cpu"
            torch_args = [f"torch==2.5.1+{channel}", f"torchvision==0.20.1+{channel}",
                          "--index-url", f"https://download.pytorch.org/whl/{channel}"]
        run(omni_pip + torch_args)
        run(omni_pip + ["-r", "requirements-omni.txt"])
    if args.mode in ("both", "misic"):
        misic_pip = prepare(misic_env, "environment-misic.yml")
        run(misic_pip + ["-r", "requirements-misic.txt"])
        # 上游包把 TensorFlow 锁在 2.4.1；本项目单独安装兼容的 TF 2.10，
        # 再安装 MiSiC 本体，避免旧依赖声明强制降级。
        run(misic_pip + ["--no-deps", "misic==1.1.5"])
    run([sys.executable, str(BASE / "main.py"), "--check", "--check-mode", args.mode])
    print("安装完成，可以提供自己的视频开始处理。")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"安装失败（退出码 {error.returncode}）。请查看上面的依赖或网络错误后重试。")
