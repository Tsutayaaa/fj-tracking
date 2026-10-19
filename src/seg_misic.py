#!/usr/bin/env python
"""第 1 步（MiSiC 线）：用 MiSiC 分割视频全部帧。

输入视频 -> mask_XXXX.npy（uint16 实例标签）

断点续传：已存在的 mask 跳过（仍逐帧读视频以对齐帧号，只省去耗时的分割）。

配方：rescale 到 STANDARD_WIDTH/MEAN_WIDTH -> add_noise(sensitivity, invert)
-> segment(invert) -> resize 回原尺寸 -> squeeze(模型输出 (H,W,1) -> 2D) -> postprocessing。

环境：conda env `misic`（MiSiC 1.1.5，Python 3.10 / TF 2.10，CPU）。
env: MISIC_MEAN_WIDTH(9.7) MISIC_SENS(0.13)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import time
import numpy as np
import cv2
from pathlib import Path
from skimage.transform import resize, rescale
from misic.extras import add_noise, postprocessing
from common import input_video, save_mask, valid_cached_mask
from model_loaders import load_misic

BASE = Path(__file__).resolve().parent.parent
VIDEO = input_video("SEG_VIDEO")
OUT = Path(os.environ.get("SEG_OUT", str(BASE / "masks_misic")))
OUT.mkdir(parents=True, exist_ok=True)

MEAN_WIDTH = float(os.environ.get("MISIC_MEAN_WIDTH", "9.7"))
SENS = float(os.environ.get("MISIC_SENS", "0.13"))
STANDARD_WIDTH = 9.7
LIGHT_BG = True


def misic_segment(g, mseg):
    sr, sc = g.shape
    scale = STANDARD_WIDTH / MEAN_WIDTH
    im = rescale(g, scale, preserve_range=True)
    img = add_noise(im, sensitivity=SENS, invert=LIGHT_BG)
    yp = mseg.segment(img, invert=LIGHT_BG)
    yp = resize(yp, (sr, sc))
    yp = np.squeeze(np.asarray(yp))
    # yp 已 resize 回原尺寸，postprocessing 必须用原图 g（不是降采样后的 im），
    # 否则 MEAN_WIDTH≠9.7 时两参数 shape 不匹配（高分辨率视频会在此崩）。
    m = postprocessing(g if LIGHT_BG else -g, yp)
    return np.asarray(m, dtype=np.uint16)


mseg = load_misic()
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
    path = OUT / f"mask_{fno:04d}.npy"
    if valid_cached_mask(path, f.shape[:2]):
        done += 1
        fno += 1
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    save_mask(path, misic_segment(g, mseg))
    done += 1
    fno += 1
    if done % 50 == 0:
        el = time.time() - t0
        print(f"seg {done}/{total}  {el:.0f}s  ETA {(total-done)*(el/max(done,1)):.0f}s", flush=True)
cap.release()
if done == 0 or (total > 0 and done != total):
    raise RuntimeError(f"视频读取不完整: {done}/{total} 帧，未继续下游分析。")
print(f"DONE {done}/{total} frames in {time.time()-t0:.1f}s", flush=True)
