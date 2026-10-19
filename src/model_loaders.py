"""从本项目 models/ 加载权重，不依赖用户目录中的模型缓存。"""
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


def model_path(name):
    path = MODELS / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型权重: {path}。请完整下载仓库中的 models/。")
    return path


def load_omni(device="auto"):
    import torch
    from cellpose_omni import models

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("OMNI_DEVICE 应为 auto、cpu 或 cuda。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但未检测到可用 GPU。请安装 GPU 版环境或使用 --device cpu。")
    use_gpu = device != "cpu" and torch.cuda.is_available()
    print(f"Omnipose 设备: {'CUDA' if use_gpu else 'CPU'}", flush=True)
    return models.CellposeModel(gpu=use_gpu,
                               pretrained_model=str(model_path("bact_phase_omnitorch_0")),
                               model_type=None, net_avg=False, nchan=2,
                               nclasses=3, omni=True)


def load_misic():
    from misic.misic import MiSiC
    from tensorflow.keras.models import load_model

    class LocalMiSiC(MiSiC):
        def __init__(self):
            # 上游构造函数无模型路径参数；初始化相同属性并保留其推理方法。
            self.model = load_model(str(model_path("MiSiCv2.h5")), compile=False)
            self.size = 256

    return LocalMiSiC()
