#!/usr/bin/env python
"""第 5 步（通用）：提特征（速度 / 物理长度 / 反转），喂聚类。

读取 tracks CSV（track_id, frame, label, area, centroid, 长短轴），逐 track 计算：
  速度 —— 质心位移 / 帧间隔（px/帧），V_MAX 以上判为离群剔除（粘连/误连跳变）
  长度 —— major_axis_length（椭圆长轴，px；有像素标定后 ×μm/px 即物理长度）
  反转 —— 连续两帧「都在动」且速度方向点积 < 0（滑动细菌的 run-reverse）

输出:
  <FEAT_TRACKS_OUT>  每条 track 的聚合特征（喂聚类）
  <FEAT_FRAMES_OUT>  逐帧时间序列（track_id/frame/speed/angle/length/width/valid）

env: FEAT_TRACKS(outputs/tracks.csv) FEAT_TRACKS_OUT(outputs/tracks_features.csv)
     FEAT_FRAMES_OUT(outputs/frames_features.csv)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRACKS = BASE / os.environ.get("FEAT_TRACKS", "outputs/tracks.csv")
TRACKS_OUT = BASE / os.environ.get("FEAT_TRACKS_OUT", "outputs/tracks_features.csv")
FRAMES_OUT = BASE / os.environ.get("FEAT_FRAMES_OUT", "outputs/frames_features.csv")

V_MAX = float(os.environ.get("FEAT_VMAX", "20.0"))  # 速度离群阈值(px/帧)，随分辨率缩放
MOVE_MIN = 1.0    # 判"在动"的速度阈值(px/帧)，低于此视为静止，不参与反转计数
REV_DOT = -0.3    # 反转判据：平滑方向夹角 dot<REV_DOT(≈>107°)，真反转≈180°

tracks = pd.read_csv(TRACKS).sort_values(["track_id", "frame"])

frame_rows = []
track_rows = []
for tid, g in tracks.groupby("track_id"):
    g = g.sort_values("frame")
    f = g["frame"].to_numpy(dtype=float)
    x = g["centroid_x"].to_numpy(dtype=float)  # 行(y)
    y = g["centroid_y"].to_numpy(dtype=float)  # 列(x)
    L = g["major_axis_length"].to_numpy(dtype=float)
    W = g["minor_axis_length"].to_numpy(dtype=float)

    dt = np.diff(f)
    dx = np.diff(y)
    dy = np.diff(x)
    speed = np.hypot(dx, dy) / np.maximum(dt, 1)
    ang = np.arctan2(dy, dx)

    valid = speed <= V_MAX

    # 反转：3 帧滑动平均速度后，连续两帧"都在动"且方向翻转 > ~107°
    def smooth(a, w=3):
        if len(a) < w:
            return a
        return np.convolve(a, np.ones(w) / w, mode="same")

    svx = smooth(dx / np.maximum(dt, 1))
    svy = smooth(dy / np.maximum(dt, 1))
    ssp = np.hypot(svx, svy)
    sang = np.arctan2(svy, svx)
    n_rev = 0
    for i in range(1, len(ssp)):
        if ssp[i - 1] > MOVE_MIN and ssp[i] > MOVE_MIN:
            if np.cos(sang[i] - sang[i - 1]) < REV_DOT:
                n_rev += 1

    # 逐帧输出（第 0 帧无速度，置 NaN）
    for i in range(len(g)):
        frame_rows.append({
            "track_id": tid,
            "frame": f[i],
            "length": L[i],
            "width": W[i],
            "speed": speed[i - 1] if i > 0 else np.nan,
            "vx": dx[i - 1] / max(dt[i - 1], 1) if i > 0 else np.nan,
            "vy": dy[i - 1] / max(dt[i - 1], 1) if i > 0 else np.nan,
            "angle": ang[i - 1] if i > 0 else np.nan,
            "valid": bool(valid[i - 1]) if i > 0 else False,
        })

    sp = speed[valid]
    duration = f[-1] - f[0] + 1
    net = np.hypot(y[-1] - y[0], x[-1] - x[0])
    # 总路程累加有效位移（px），跨帧漏检时保留完整位移。
    path = np.hypot(dx[valid], dy[valid]).sum()
    ar = L / np.maximum(W, 1)
    track_rows.append({
        "track_id": tid,
        "duration": duration,
        "n_det": len(g),
        "speed_mean": sp.mean() if len(sp) else np.nan,
        "speed_median": np.median(sp) if len(sp) else np.nan,
        "speed_max": sp.max() if len(sp) else np.nan,
        "speed_std": sp.std() if len(sp) else np.nan,
        "speed_p90": np.percentile(sp, 90) if len(sp) else np.nan,
        "length_mean": L.mean(),
        "length_median": np.median(L),
        "length_std": L.std(),
        "width_mean": W.mean(),
        "aspect_ratio_mean": ar.mean(),
        "net_displacement": net,
        "path_length": path,
        "persistence": net / path if path > 0 else np.nan,
        "n_reversals": n_rev,
        "reversal_rate": n_rev / (duration / 1.0) if duration > 0 else np.nan,
        "n_outlier": int((~valid).sum()),
    })

tf = pd.DataFrame(track_rows, columns=[
    "track_id", "duration", "n_det", "speed_mean", "speed_median", "speed_max",
    "speed_std", "speed_p90", "length_mean", "length_median", "length_std",
    "width_mean", "aspect_ratio_mean", "net_displacement", "path_length",
    "persistence", "n_reversals", "reversal_rate", "n_outlier",
]).sort_values("track_id")
ff = pd.DataFrame(frame_rows, columns=[
    "track_id", "frame", "length", "width", "speed", "vx", "vy", "angle", "valid",
]).sort_values(["track_id", "frame"])
TRACKS_OUT.parent.mkdir(parents=True, exist_ok=True)
FRAMES_OUT.parent.mkdir(parents=True, exist_ok=True)
tf.to_csv(TRACKS_OUT, index=False)
ff.to_csv(FRAMES_OUT, index=False)

print(f"track 特征: {len(tf)} 条 -> {TRACKS_OUT}")
print(f"逐帧特征: {len(ff)} 条 -> {FRAMES_OUT}")
# 摘要
long = tf[tf["duration"] >= 20]
print(f"\n时长>=20 帧的 track: {len(long)} 条")
if long.empty:
    raise SystemExit(0)
print(f"  速度(px/帧): 中位 {long['speed_median'].median():.2f}, 全样本中位 {np.nanmedian(long['speed_mean']):.2f}")
print(f"  长度(px):    中位 {long['length_mean'].median():.1f}, 跨度 {long['length_mean'].min():.0f}-{long['length_mean'].max():.0f}")
print(f"  反转率(次/帧): 中位 {long['reversal_rate'].median():.4f}")
print(f"  持久性 net/path: 中位 {long['persistence'].median():.3f}")
