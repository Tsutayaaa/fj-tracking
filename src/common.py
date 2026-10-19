"""独立步骤共用的路径、mask 和视频检查。"""
import os
from pathlib import Path

import numpy as np


def input_video(variable):
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"请通过 main.py 运行，或设置环境变量 {variable}。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"输入视频不存在: {path}")
    return path


def mask_files(folder):
    files = sorted(Path(folder).glob("mask_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise RuntimeError(f"未找到分割 mask: {folder}")
    frames = [int(p.stem.split("_")[-1]) for p in files]
    if frames != list(range(len(files))):
        raise RuntimeError(f"mask 帧号缺失或重复，请先重新完成分割: {folder}")
    return files


def save_mask(path, mask):
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(mask, dtype=np.uint16))
    temporary.replace(path)


def valid_cached_mask(path, shape):
    if not path.exists():
        return False
    try:
        mask = np.load(path, mmap_mode="r", allow_pickle=False)
        if mask.shape != shape or mask.dtype != np.uint16:
            raise ValueError("mask 尺寸或类型不符")
    except (OSError, ValueError, EOFError) as error:
        raise RuntimeError(f"缓存 mask 损坏或不匹配，请删除此文件后重跑: {path}") from error
    return True
