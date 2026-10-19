# 细菌追踪管线

输入用户自己的相差显微视频，完成细胞分割、追踪、特征提取和追踪动画输出。支持 Omnipose 和 MiSiC 两个分割后端。

仓库包含源码、运行环境配置和两份模型权重，不包含示例视频、实验数据、运行结果或缓存。

## 安装

先安装 Miniconda，并确保当前终端能运行 `conda`。进入项目根目录后执行：

```bash
python install.py
```

安装脚本创建两个 Python 3.10 环境：`omnipose` 用于 Omnipose 及全部下游处理，`misic` 用于 MiSiC 分割。默认安装 CPU 版，已有同名 Python 3.10 环境会安装或调整为本项目要求的依赖版本；如需保留原环境，可先通过 `OMNI_ENV`、`MISIC_ENV` 环境变量指定其他名称。

Linux/Windows 的 NVIDIA GPU 用户可选择 CUDA 12.1 版 PyTorch，显卡驱动需兼容该版本：

```bash
python install.py --cuda
```

仅安装一个分割后端时，可在安装命令后指定 `omni` 或 `misic`。MiSiC 单独安装时，仍会准备 `omnipose` 环境中的下游处理依赖，但不安装 Omnipose 模型依赖。

`environment.yml` 和 `environment-misic.yml` 定义基础 Python 环境，依赖由安装脚本继续安装；不要只创建基础环境就直接运行分析。MiSiC 1.1.5 的上游依赖声明锁定了旧版 TensorFlow 2.4.1，安装脚本先安装 TF 2.10，再以 `--no-deps` 安装 MiSiC 本体，以适配 Python 3.10。Apple Silicon 使用 `tensorflow-macos`，其他受支持平台使用 `tensorflow`。

安装需要联网下载依赖。Linux ARM 等缺少 TensorFlow 2.10 对应发行包的平台未列为本项目的支持目标。

## 检查安装

安装脚本最后自动检查依赖，并实际加载两份本地权重。之后也可以手动检查：

```bash
python main.py --check
```

仅检查一个后端时，使用 `--check --check-mode omni` 或 `--check --check-mode misic`。

## 运行

命令格式：

```text
python main.py <视频路径> [both|omni|misic] [--device auto|cpu|cuda] [--output-dir <结果目录>]
```

视频由用户自行提供，可放在任意可读路径；包含空格的路径需要引号。省略后端时运行两条分割线。Omnipose 默认优先使用可用的 NVIDIA CUDA，否则使用 CPU；MiSiC 使用 CPU。启动入口只依赖 Python 标准库，不必先激活某个处理环境。

`--output-dir` 指定本次视频的完整结果目录；省略时写入项目下的 `outputs/<视频名>/`。其他参数可用 `python main.py --help` 查看。

## 输出

每个后端输出以下文件，`{backend}` 为 `omni` 或 `misic`：

| 文件 | 内容 |
|---|---|
| `tracks_animation_{backend}.mp4` | 细胞填充和完整质心轨迹动画 |
| `tracks_{backend}.csv` | 检测、轨迹编号、帧号、形态和位置 |
| `tracks_features_{backend}.csv` | 每条轨迹的速度、形态、路程、反转等特征 |
| `frames_features_{backend}.csv` | 每次检测的形态和运动特征 |
| `run_manifest.json` | 输入内容、参数及模型校验值，用于保护断点续跑 |

中间 mask 也写入结果目录。检测不到符合面积条件的细胞时，输出具有固定表头的空表和原画面动画。

重新运行相同输入、参数和权重时，会复用完整的分割 mask，并重算下游结果。如果内容或配置改变，入口会要求使用新的输出目录，防止把旧 mask 当成新结果。中断写入的 mask 使用临时文件；已存在但损坏的 mask 会明确报错，删除所提示的文件后可重跑。

## 模型

两份权重随仓库提供，脚本直接加载，无需手动复制到用户模型缓存或第三方包目录：

| 路径 | 用途 |
|---|---|
| `models/bact_phase_omnitorch_0` | Omnipose 相差细菌分割 |
| `models/MiSiCv2.h5` | MiSiC 分割 |
| `models/SHA256SUMS` | 权重文件的 SHA-256 校验值 |

请完整下载仓库中的模型文件，不要用 GitHub 网页的 HTML 内容代替权重。

## 参数与指标口径

默认参数延续原项目：以视频宽度 1824 像素为基准缩放细胞宽度、最小面积和匹配距离。该假设适用于同一成像条件下的等比例缩放；裁剪或改变显微倍率时需重新校准参数。底层步骤的环境变量可在各脚本中查看。

速度单位为像素/帧，长度单位为像素，时长单位为帧，反转率单位为次/帧。当前没有自动转换到微米/秒，也没有使用视频的逐帧时间戳；不等时间间隔采集的数据需要额外处理后再做物理速度比较。

总路程已修正为有效相邻观测位移之和，跨帧漏检时不会把速度误当成路程。缺失区间只能计算观测端点的直线位移，不能恢复期间的真实绕行。异常位移被排除在路程之外，但净位移仍使用整条轨迹首尾，因此包含异常段时 `persistence` 仍可能大于 1，使用时应结合 `n_outlier` 筛选。反转计数保持原项目的方向平滑与阈值算法，并非经过人工标注验证的事件检测器。

为兼容已有输出，检测表中的 `centroid_x` 表示行坐标（图像 y），`centroid_y` 表示列坐标（图像 x）。`major_axis_length` 是等效椭圆长轴，不直接等于弯曲细胞的中心线长度。

## 上游项目

- [Omnipose](https://github.com/kevinjohncutler/omnipose) 与其 [模型加载接口](https://omnipose.readthedocs.io/settings.html)
- [MiSiC 1.1.5](https://pypi.org/project/misic/1.1.5/)
- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)

本仓库保留原项目的分割、拆分和追踪策略；整理交付不等于已完成科研准确性验证。

