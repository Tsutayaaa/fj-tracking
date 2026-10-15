# fj-tracking

[English](README.md) | **简体中文**

本项目是 [ZhuLab](https://connect.xjtlu.edu.cn/user/zhulab.) 的研究项目，用于从显微视频中追踪 *Flavobacterium johnsoniae*（约翰逊黄杆菌）的运动，并开展定量分析。

## 项目状态

项目目前处于初始化阶段，尚未实现视频处理、细胞追踪及定量分析功能。目录结构与依赖将随开发需要逐步添加。

## 工作区设置

当前入口仅使用 Python 标准库，可在 Python 3.9 及以上版本运行。

```bash
git clone https://github.com/Tsutayaaa/fj-tracking.git
cd fj-tracking
python3 -m venv .venv
```

在 macOS / Linux 中激活环境：

```bash
source .venv/bin/activate
```

在 Windows 命令提示符中激活环境：

```bat
.venv\Scripts\activate.bat
```

在 PyCharm 中打开仓库，将项目解释器设为 `.venv`。运行以下命令检查入口：

```bash
python main.py
```

目前入口仅输出项目状态。

## 工作区约定

- `.editorconfig` 统一 UTF-8 编码、LF 换行及 Python 四空格缩进。
- `.gitignore` 排除本地虚拟环境、IDE 设置、Python 缓存、构建产物和本地环境变量文件。
- 源代码目录、数据组织方式及依赖配置按后续实际需要添加。

## 许可证

项目尚未指定开源许可证。
