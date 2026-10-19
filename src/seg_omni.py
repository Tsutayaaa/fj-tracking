#!/usr/bin/env python
"""第 1 步（omni 线）：用 Omnipose `bact_phase_omni` 分割视频全部帧。

输入视频 -> mask_XXXX.npy（uint16 实例标签，0=背景，1..N=实例）

断点续传：已存在的 mask 跳过（仍逐帧读视频以对齐帧号，只省去耗时的分割）。

直接加载项目 models/ 中的权重。OMNI_DEVICE=auto|cpu|cuda。
"""
import os
import time
import numpy as np
import cv2
from pathlib import Path
from omnipose.utils import normalize99
from common import input_video, save_mask, valid_cached_mask
from model_loaders import load_omni

BASE = Path(__file__).resolve().parent.parent
VIDEO = input_video("SEG_VIDEO")
MASK_DIR = Path(os.environ.get("SEG_OUT", str(BASE / "masks_omni")))
MASK_DIR.mkdir(parents=True, exist_ok=True)

params = dict(channels=[0, 0], mask_threshold=-1, flow_threshold=0,
              omni=True, cluster=True, resample=True, verbose=False,
              tile=False, niter=7, augment=False)
model = load_omni(os.environ.get("OMNI_DEVICE", "auto"))

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO}")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
t0 = time.time()
done = 0
fno = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    path = MASK_DIR / f"mask_{fno:04d}.npy"
    if valid_cached_mask(path, f.shape[:2]):
        done += 1
        fno += 1
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    m, _, _ = model.eval(normalize99(g), **params)
    save_mask(path, m)
    done += 1
    fno += 1
    if fno % 100 == 0:
        print(f"segmented {fno}/{total} ({time.time()-t0:.0f}s)", flush=True)
cap.release()
if done == 0 or (total > 0 and done != total):
    raise RuntimeError(f"视频读取不完整: {done}/{total} 帧，未继续下游分析。")
print(f"DONE {done} frames in {time.time()-t0:.1f}s = {(time.time()-t0)/max(done,1):.3f} s/frame",
      flush=True)
