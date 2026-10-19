#!/usr/bin/env python
"""细菌视频分割、追踪和特征提取。运行 python main.py --help 查看参数。"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"
OMNI_ENV = os.environ.get("OMNI_ENV", "omnipose")
MISIC_ENV = os.environ.get("MISIC_ENV", "misic")


def conda_command(env_name):
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda or not Path(conda).is_file():
        raise RuntimeError("未找到 conda。请安装 Miniconda，并先运行 python install.py。")
    return [conda, "run", "--no-capture-output", "-n", env_name, "python"]


def step(conda_env, script, **envvars):
    env = dict(os.environ)
    env.update({key: str(value) for key, value in envvars.items()})
    print(f"+ {script} [{conda_env}]", flush=True)
    subprocess.run(conda_command(conda_env) + [str(SRC / script)],
                   cwd=str(BASE), env=env, check=True)


def check_installation(mode, load_models=False):
    backend = "omni" if mode in ("both", "omni") else "downstream"
    step(OMNI_ENV, "check_runtime.py", CHECK_BACKEND=backend,
         CHECK_LOAD_MODELS=int(load_models))
    if mode in ("both", "misic"):
        step(MISIC_ENV, "check_runtime.py", CHECK_BACKEND="misic",
             CHECK_LOAD_MODELS=int(load_models))


def video_shape_fps(video):
    # 在处理环境内读取，不要求启动 main.py 的 Python 安装 OpenCV。
    code = (
        "import cv2,json,sys; c=cv2.VideoCapture(sys.argv[1]); "
        "assert c.isOpened(), '无法打开输入视频'; "
        "print(json.dumps([int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),"
        "int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),c.get(cv2.CAP_PROP_FPS)])); c.release()"
    )
    result = subprocess.run(conda_command(OMNI_ENV) + ["-c", code, str(video)],
                            capture_output=True, text=True, check=True)
    width, height, fps = json.loads(result.stdout.strip().splitlines()[-1])
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("视频尺寸或帧率无效，请检查输入文件。")
    return width, height, fps


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(folder, video, parameters):
    manifest = folder / "run_manifest.json"
    record = {"video_sha256": sha256(video), "parameters": parameters,
              "models": {p.name: sha256(p) for p in sorted((BASE / "models").iterdir())
                         if p.is_file() and p.name != "SHA256SUMS"}}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != record:
            raise RuntimeError("输出目录对应的视频、参数或权重已改变。请用 --output-dir 指定新目录，避免混用旧 mask。")
    elif folder.exists() and any(folder.iterdir()):
        raise RuntimeError("输出目录已有无运行记录的文件。请用 --output-dir 指定新的空目录。")
    folder.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finish_pipeline(video, folder, backend, masks, minarea, maxdist, vmax, fps):
    clean = folder / f"masks_{backend}_clean"
    tracks = folder / f"tracks_{backend}.csv"
    step(OMNI_ENV, "clean_masks.py", CLN_SRC=masks, CLN_OUT=clean,
         CLN_DUP="", CLN_MIN_AREA=minarea)
    step(OMNI_ENV, "track_sort.py", SORT_MASK_DIR=clean, SORT_OUT=tracks,
         SORT_MAX_DIST=maxdist)
    step(OMNI_ENV, "extract_features.py", FEAT_TRACKS=tracks,
         FEAT_TRACKS_OUT=folder / f"tracks_features_{backend}.csv",
         FEAT_FRAMES_OUT=folder / f"frames_features_{backend}.csv", FEAT_VMAX=vmax)
    step(OMNI_ENV, "animate_tracks.py", ANIM_VIDEO=video, ANIM_MASKS=clean,
         ANIM_TRACKS=tracks, ANIM_OUT=folder / f"tracks_animation_{backend}.mp4",
         ANIM_DUP="", ANIM_FPS=fps)


def run_omni(video, folder, minarea, maxdist, vmax, fps, device):
    step(OMNI_ENV, "seg_omni.py", SEG_VIDEO=video, SEG_OUT=folder / "masks_omni",
         OMNI_DEVICE=device)
    step(OMNI_ENV, "split_merged.py", SPLIT_IN=folder / "masks_omni",
         SPLIT_OUT=folder / "masks_omni_split", SPLIT_MIN_AREA=minarea)
    finish_pipeline(video, folder, "omni", folder / "masks_omni_split", minarea, maxdist, vmax, fps)


def run_misic(video, folder, mew, minarea, maxdist, vmax, fps):
    step(MISIC_ENV, "seg_misic.py", SEG_VIDEO=video, SEG_OUT=folder / "masks_misic",
         MISIC_MEAN_WIDTH=mew)
    finish_pipeline(video, folder, "misic", folder / "masks_misic", minarea, maxdist, vmax, fps)


def main():
    parser = argparse.ArgumentParser(description="细菌视频分割、追踪和特征提取")
    parser.add_argument("video", nargs="?", type=Path, help="用户提供的输入视频路径")
    parser.add_argument("mode", nargs="?", choices=("both", "omni", "misic"), default="both")
    parser.add_argument("--check", action="store_true", help="检查环境并实际加载模型，不处理视频")
    parser.add_argument("--check-mode", choices=("both", "omni", "misic"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Omnipose 设备；auto 优先 CUDA，无 CUDA 时使用 CPU")
    parser.add_argument("--output-dir", type=Path, help="本次视频的结果目录")
    args = parser.parse_args()
    if args.check:
        check_installation(args.check_mode, load_models=True)
        print("安装检查通过。")
        return
    if args.video is None:
        parser.error("请指定输入视频，或使用 --check 检查安装。")
    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"找不到输入视频: {video}")
    check_installation(args.mode)
    width, height, fps = video_shape_fps(video)
    scale = width / 1824.0
    mew = round(9.7 * scale, 2)
    minarea = max(1, round(100 * scale * scale))
    maxdist = max(1, round(30 * scale))
    vmax = max(1, round(20 * scale))
    folder = args.output_dir.expanduser().resolve() if args.output_dir else BASE / "outputs" / video.stem
    prepare_output(folder, video, {"mean_width": mew, "min_area": minarea,
                                  "max_distance": maxdist, "vmax": vmax})
    print(f"输入 {video} ({width}x{height}, fps={fps:.3f})", flush=True)
    if args.mode in ("both", "omni"):
        run_omni(video, folder, minarea, maxdist, vmax, fps, args.device)
    if args.mode in ("both", "misic"):
        run_misic(video, folder, mew, minarea, maxdist, vmax, fps)
    print(f"完成 -> {folder}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(f"步骤执行失败（退出码 {error.returncode}）。请检查上述错误及 README.md。")
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))
