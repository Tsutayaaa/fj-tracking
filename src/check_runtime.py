"""检查当前环境依赖及随附模型；由 main.py --check 调用。"""
import os
import sys

try:
    backend = os.environ.get("CHECK_BACKEND", "downstream")
    if backend in ("omni", "downstream"):
        import cv2
        import matplotlib
        import numpy
        import pandas
        import scipy
        import skimage
    if backend == "omni":
        import torch
        from cellpose_omni import models
        from omnipose.utils import normalize99
        from model_loaders import load_omni, model_path
        model_path("bact_phase_omnitorch_0")
        if os.environ.get("CHECK_LOAD_MODELS") == "1":
            load_omni("cpu")
    elif backend == "misic":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        import cv2
        import tensorflow
        from misic.misic import MiSiC
        from misic.extras import add_noise, postprocessing
        from model_loaders import load_misic, model_path
        model_path("MiSiCv2.h5")
        if os.environ.get("CHECK_LOAD_MODELS") == "1":
            load_misic()
    print(f"OK: {backend} (Python {sys.version.split()[0]})", flush=True)
except Exception as error:
    raise SystemExit(f"环境检查失败: {error}\n请按 README.md 重新安装对应环境。") from error
