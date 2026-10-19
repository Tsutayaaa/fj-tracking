#!/usr/bin/env python
"""第 2 步（omni 线）：时间一致性 force-split。

Omnipose `cluster=True` 切不开的硬粘连（滑动细菌擦肩而过），靠相邻帧时间信息切：
若帧 f 的实例 M 在帧 f-1 有两个前驱 A,B（IoU>MERGE_IOU，即 A,B 都被并进了 M），
说明 M 是 A、B 的粘连体 —— 用 A、B 的质心做 watershed 种子把 M 切回两个细胞。

前向传递：每处理完一帧就落盘，下一帧以「已切好」的前一帧做参照，
所以跨多帧的持续粘连会逐帧自动切。内含去碎屑(area<MIN_AREA)。

masks_omni/mask_XXXX.npy -> masks_omni_split/mask_XXXX.npy
"""
import os
import numpy as np
from pathlib import Path
from skimage import measure
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from common import mask_files, save_mask

BASE = Path(__file__).resolve().parent.parent
MASK_DIR = Path(os.environ.get("SPLIT_IN", str(BASE / "masks_omni")))
OUT = Path(os.environ.get("SPLIT_OUT", str(BASE / "masks_omni_split")))
OUT.mkdir(parents=True, exist_ok=True)

MERGE_IOU = 0.3      # 前驱与粘连体的 IoU 下限（A⊂M 时 ≈0.5）
MIN_AREA = float(os.environ.get("SPLIT_MIN_AREA", "100"))  # 碎屑面积阈值（随分辨率缩放）


def clean_mask(m, min_area=MIN_AREA):
    if m.max() == 0:
        return m
    areas = np.bincount(m.ravel())
    keep = np.zeros(areas.size, dtype=bool)
    keep[1:] = areas[1:] >= min_area
    remap = np.zeros(areas.size, dtype=np.uint16)
    remap[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint16)
    return remap[m]


def iou_mat(m1, m2):
    n1 = m1.max() + 1
    n2 = m2.max() + 1
    idx = m1.ravel().astype(np.int64) * n2 + m2.ravel().astype(np.int64)
    ov = np.bincount(idx, minlength=n1 * n2).reshape(n1, n2)
    a1 = np.bincount(m1.ravel(), minlength=n1)
    a2 = np.bincount(m2.ravel(), minlength=n2)
    u = a1[:, None] + a2[None, :] - ov
    iou = np.divide(ov, u, out=np.zeros_like(ov, dtype=float), where=u > 0)
    iou[0, :] = 0
    iou[:, 0] = 0
    return iou


def centroids(m):
    r = measure.regionprops_table(m, properties=["label", "centroid"])
    return {int(l): (cy, cx) for l, cy, cx in
            zip(r["label"], r["centroid-0"], r["centroid-1"])}


def split_instance(m, label, seeds):
    """把 m 中实例 label 按两个种子点 watershed 切成 2 个，返回 {1,2} 的子掩码(整帧)。"""
    binary = (m == label)
    dist = ndi.distance_transform_edt(binary)
    markers = np.zeros(m.shape, int)
    for k, (y, x) in enumerate(seeds, start=1):
        yi, xi = int(round(y)), int(round(x))
        if not binary[yi, xi]:
            # 种子落到粘连体外，投影到最近前景点
            yy, xx = np.nonzero(binary)
            j = np.argmin((yy - y) ** 2 + (xx - x) ** 2)
            yi, xi = yy[j], xx[j]
        markers[yi, xi] = k
    return watershed(-dist, markers, mask=binary)


def main():
    files = mask_files(MASK_DIR)
    prev = clean_mask(np.load(files[0]))
    save_mask(OUT / "mask_0000.npy", prev)

    n_split = 0
    n_skip3 = 0
    for f in range(1, len(files)):
        cur = clean_mask(np.load(files[f]))
        iou = iou_mat(prev, cur)
        prev_cent = centroids(prev)

        to_split = {}
        for c in range(1, cur.max() + 1):
            preds = [int(p) for p in np.where(iou[:, c] > MERGE_IOU)[0] if p != 0]
            if len(preds) == 2:
                to_split[c] = [prev_cent[p] for p in preds]
            elif len(preds) >= 3:
                n_skip3 += 1

        if not to_split:
            new_m = cur
        else:
            new_m = np.zeros_like(cur, dtype=np.uint16)
            nl = 0
            for c in range(1, cur.max() + 1):
                if c in to_split:
                    ws = split_instance(cur, c, to_split[c])
                    for sub in (1, 2):
                        sel = (ws == sub)
                        if sel.any():
                            nl += 1
                            new_m[sel] = nl
                    n_split += 1
                else:
                    sel = (cur == c)
                    if sel.any():
                        nl += 1
                        new_m[sel] = nl

        prev = new_m
        save_mask(OUT / f"mask_{f:04d}.npy", new_m)
        if f % 100 == 0:
            print(f"split {f}/{len(files)} 累计切 {n_split} 起(3+前驱跳过 {n_skip3})", flush=True)

    print(f"DONE 共切粘连 {n_split} 起，3+前驱跳过 {n_skip3}", flush=True)


if __name__ == "__main__":
    main()
