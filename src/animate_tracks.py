#!/usr/bin/env python
"""第 6 步（通用）：渲染轨迹动画（完整轨迹保留 + 细胞填充）。

风格：所有 track，每个细胞用其 track 颜色：
  1) 完整轨迹：从 track 起点到当前帧的整条质心路径，随时间生长、全程保留，gap 断开
  2) 当前帧细胞半透明填充 + 描轮廓
  3) 当前质心圆点（轨迹"头"）

帧对齐：若源视频含重复帧（ANIM_DUP 传入帧号），逐帧顺序读取并跳过，输出帧数与
去重后的 track frame 索引一一对应。

坐标：centroid_x=行(y)、centroid_y=列(x)；图像点=(x=centroid_y, y=centroid_x)。

env: ANIM_VIDEO(必填) ANIM_MASKS(masks_misic_clean)
     ANIM_TRACKS(outputs/tracks.csv) ANIM_OUT(outputs/tracks_animation.mp4)
"""
import os
import numpy as np
import cv2
import pandas as pd
from bisect import bisect_right
from pathlib import Path
from matplotlib import colormaps
from common import input_video, mask_files

BASE = Path(__file__).resolve().parent.parent
VIDEO = input_video("ANIM_VIDEO")
CSV = BASE / os.environ.get("ANIM_TRACKS", "outputs/tracks.csv")
MASKS = BASE / os.environ.get("ANIM_MASKS", "masks_misic_clean")
OUT = BASE / os.environ.get("ANIM_OUT", "outputs/tracks_animation.mp4")

_dup = os.environ.get("ANIM_DUP", "")
DUP = {int(x) for x in _dup.split(",") if x.strip() != ""}
TRAIL_LINE = 1                                           # 轨迹线宽
ALPHA = 0.40                                             # 细胞填充 alpha
FPS = float(os.environ.get("ANIM_FPS", "11.59"))

tracks = pd.read_csv(CSV)
ids = sorted(tracks["track_id"].unique())
print(f"total tracks={len(ids)}", flush=True)

# per-track：sorted arrays  frame / col / row；label map
trails = {}   # pid -> (frames[], cols[], rows[])
labmap = {}   # pid -> {frame: label}
for pid in ids:
    g = tracks[tracks.track_id == pid].sort_values("frame")
    trails[pid] = (g["frame"].to_numpy(),
                   g["centroid_y"].to_numpy(),   # col = x
                   g["centroid_x"].to_numpy())   # row = y
    labmap[pid] = {int(r.frame): int(r.label) for r in g.itertuples(index=False)}

# 预计算每个 track 的连续段（gap：frame 差 > 1 处断开）-> [(start_idx, end_idx))
runs = {}
for pid in ids:
    fr = trails[pid][0]
    starts = [0]
    for k in range(1, len(fr)):
        if fr[k] - fr[k - 1] > 1:
            starts.append(k)
    starts.append(len(fr))
    runs[pid] = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]

cm = colormaps["hsv"]
def _rgb(i):
    c = cm(i / max(len(ids), 1))
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
bgr = {pid: _rgb(k)[::-1] for k, pid in enumerate(ids)}   # -> BGR

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
KEEP = [i for i in range(TOTAL) if i not in DUP]   # 去重帧号 -> 原始帧号（无重复帧时即 0..TOTAL-1）
if len(mask_files(MASKS)) != len(KEEP):
    cap.release()
    raise RuntimeError("mask 数量与视频帧数不一致，请先完成分割和清理。")
OUT.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"无法创建输出视频，请检查路径和 MP4 编码支持: {OUT}")

f = 0   # 去重帧号
r = 0   # 原始帧号
while f < len(KEEP):
    ok, frame = cap.read()
    if not ok:
        break
    if r in DUP:
        r += 1
        continue
    mask = np.load(MASKS / f"mask_{f:04d}.npy")

    # 1) 完整轨迹（底层）：到当前帧为止的整条路径，gap 断开，全程保留
    for pid in ids:
        fr, cx, cy = trails[pid]
        upto = bisect_right(fr, f)          # frame <= f 的点数
        if upto < 2:
            continue
        pts = np.column_stack([cx, cy]).astype(np.int32)
        for (s, e) in runs[pid]:
            if s >= upto:
                break
            e2 = min(e, upto)
            if e2 - s >= 2:
                cv2.polylines(frame, [pts[s:e2].reshape(-1, 1, 2)],
                              False, bgr[pid], TRAIL_LINE, cv2.LINE_AA)

    # 2) 当前帧细胞填充 + 轮廓 + 质心圆点（上层）
    for pid in ids:
        lab = labmap[pid].get(f)
        if lab is None:
            continue
        cell = (mask == lab)
        if cell.any():
            c = bgr[pid]
            frame[cell] = (frame[cell].astype(np.float32) * (1 - ALPHA)
                           + np.array(c, np.float32) * ALPHA).astype(np.uint8)
            cnts, _ = cv2.findContours(cell.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, c, 2)
            fr, cx, cy = trails[pid]
            idx = np.searchsorted(fr, f)
            if idx < len(fr) and fr[idx] == f:
                cv2.circle(frame, (int(cx[idx]), int(cy[idx])), 2, c, -1)

    cv2.putText(frame, f"tracks  frame {f}/{len(KEEP)}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    writer.write(frame)
    if f % 100 == 0:
        print(f"  {f}/{len(KEEP)}", flush=True)
    f += 1
    r += 1

cap.release()
writer.release()
if f != len(KEEP):
    raise RuntimeError(f"视频解码提前结束，动画不完整: {f}/{len(KEEP)} 帧。")
print(f"saved {OUT}", flush=True)
print("DONE", flush=True)
