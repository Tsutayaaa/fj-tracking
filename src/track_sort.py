#!/usr/bin/env python
"""第 4 步（通用）：SORT 式追踪（质心距离 + Hungarian + 记忆）。

对分割 flicker 鲁棒：质心距离不受 mask 形状抖动影响；记忆机制允许检测
短暂丢失（最多 MEMORY 帧）后按预测位置找回，直接治住 orphan/split-loser 碎片。

读取 *_clean/mask_XXXX.npy，输出 tracks CSV（track_id, frame, label, area,
centroid_x, centroid_y, major_axis_length, minor_axis_length）。

env: SORT_MASK_DIR(masks_misic_clean) SORT_OUT(outputs/tracks.csv)
     SORT_MAX_DIST(30) SORT_MEMORY(15) SORT_VW(0)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure
from scipy.optimize import linear_sum_assignment
from common import mask_files

BASE = Path(__file__).resolve().parent.parent
MASK_DIR = BASE / os.environ.get("SORT_MASK_DIR", "masks_misic_clean")
OUT_CSV = BASE / os.environ.get("SORT_OUT", "outputs/tracks.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

MAX_DIST = float(os.environ.get("SORT_MAX_DIST", "30.0"))  # 匹配门限(px)：中位跳 7.7、p90 21.9，30 避开可疑尾
MEMORY = int(os.environ.get("SORT_MEMORY", "15"))          # 允许连续丢失帧数：覆盖 ~90% gap
V_WEIGHT = float(os.environ.get("SORT_VW", "0.0"))         # 恒速外推权重：0=纯质心距离（外推被粘连污染）


def regionprops(m):
    r = measure.regionprops_table(
        m, properties=["label", "area", "centroid", "major_axis_length", "minor_axis_length"])
    return [{
        "label": int(r["label"][i]),
        "area": float(r["area"][i]),
        "y": float(r["centroid-0"][i]),
        "x": float(r["centroid-1"][i]),
        "major": float(r["major_axis_length"][i]),
        "minor": float(r["minor_axis_length"][i]),
    } for i in range(len(r["label"]))]


def main():
    files = mask_files(MASK_DIR)
    n = len(files)
    active = []     # 活跃 track：id/cx/cy/vx/vy/missed
    next_id = 0
    rows = []

    for f in range(n):
        dets = regionprops(np.load(files[f]))

        if not active:
            for d in dets:
                active.append({"id": next_id, "cx": d["x"], "cy": d["y"],
                               "vx": 0.0, "vy": 0.0, "missed": 0})
                rows.append((next_id, f, d))
                next_id += 1
            continue

        # 恒速预测每个 track 的当前位置
        pred = np.array([(t["cx"] + V_WEIGHT * t["vx"], t["cy"] + V_WEIGHT * t["vy"])
                         for t in active])
        det_pos = np.array([(d["x"], d["y"]) for d in dets]) if dets else np.zeros((0, 2))
        if len(dets):
            D = np.linalg.norm(pred[:, None, :] - det_pos[None, :, :], axis=2)
            D[D > MAX_DIST] = 1e9
            ri, ci = linear_sum_assignment(D)
            assigned_t = set()
            assigned_d = set()
            for r, c in zip(ri, ci):
                if D[r, c] <= MAX_DIST:
                    t, d = active[r], dets[c]
                    t["vx"] = d["x"] - t["cx"]   # 用实际位移更新速度
                    t["vy"] = d["y"] - t["cy"]
                    t["cx"], t["cy"] = d["x"], d["y"]
                    t["missed"] = 0
                    rows.append((t["id"], f, d))
                    assigned_t.add(r)
                    assigned_d.add(c)
            # 未匹配 track -> missed
            for r in range(len(active)):
                if r not in assigned_t:
                    active[r]["missed"] += 1
            # 未匹配检测 -> 新 track
            for c in range(len(dets)):
                if c not in assigned_d:
                    d = dets[c]
                    active.append({"id": next_id, "cx": d["x"], "cy": d["y"],
                                   "vx": 0.0, "vy": 0.0, "missed": 0})
                    rows.append((next_id, f, d))
                    next_id += 1
        else:
            for t in active:
                t["missed"] += 1

        # 终结丢失过久的 track
        active = [t for t in active if t["missed"] <= MEMORY]

    df = pd.DataFrame([{
        "track_id": tid, "frame": f, "label": d["label"], "area": d["area"],
        "centroid_x": d["y"], "centroid_y": d["x"],
        "major_axis_length": d["major"], "minor_axis_length": d["minor"],
    } for (tid, f, d) in rows], columns=[
        "track_id", "frame", "label", "area", "centroid_x", "centroid_y",
        "major_axis_length", "minor_axis_length",
    ]).sort_values(["track_id", "frame"])
    df.to_csv(OUT_CSV, index=False)

    if df.empty:
        print(f"未检测到符合条件的细胞，已写入空表: {OUT_CSV}")
        return

    lens = df.groupby("track_id").size()
    print(f"track 数: {df['track_id'].nunique()}")
    print(f"帧长: 中位 {lens.median():.0f} | "
          f">=20帧 {lens[lens >= 20].sum() / len(df) * 100:.1f}% 检测 | "
          f">=50帧 {lens[lens >= 50].sum() / len(df) * 100:.1f}% | "
          f">=100帧 {lens[lens >= 100].sum() / len(df) * 100:.1f}%")
    print(f"已写 {OUT_CSV}")


if __name__ == "__main__":
    main()
