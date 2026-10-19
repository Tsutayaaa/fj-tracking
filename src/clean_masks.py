#!/usr/bin/env python
"""第 3 步（通用）：去碎屑(area<MIN_AREA) + 去重复帧 -> *_clean/，重编号 0..N-1。

可选：若源视频含重复帧，用 CLN_DUP 传入逗号分隔的帧号将其去掉；CLN_DUP="" 表示无重复帧。
omni 线与 MiSiC 线共用本步，保证两条线下游帧索引与口径一致，可公平对比。

omni 线:  masks_omni_split -> masks_omni_clean
MiSiC 线: masks_misic      -> masks_misic_clean

env: CLN_SRC(masks_misic) CLN_OUT(masks_misic_clean) CLN_MIN_AREA(100)
"""
import os
import numpy as np
from pathlib import Path
from common import mask_files, save_mask

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / os.environ.get("CLN_SRC", "masks_misic")
OUT = BASE / os.environ.get("CLN_OUT", "masks_misic_clean")
OUT.mkdir(parents=True, exist_ok=True)
MIN_AREA = float(os.environ.get("CLN_MIN_AREA", "100"))

# 重复帧索引（默认无重复帧；若源视频含重复帧，用 CLN_DUP 传入逗号分隔的帧号）
_dup = os.environ.get("CLN_DUP", "")
DUP = {int(x) for x in _dup.split(",") if x.strip() != ""}


def clean(m):
    if m.max() == 0:
        return m
    areas = np.bincount(m.ravel())
    keep = np.zeros(areas.size, dtype=bool)
    keep[1:] = areas[1:] >= MIN_AREA
    remap = np.zeros(areas.size, dtype=np.uint16)
    remap[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint16)
    return remap[m]


files = mask_files(SRC)
keep = [i for i in range(len(files)) if i not in DUP]
for new, old in enumerate(keep):
    save_mask(OUT / f"mask_{new:04d}.npy", clean(np.load(files[old], allow_pickle=False)))
print(f"DONE {len(keep)} 帧 -> {OUT}/")
