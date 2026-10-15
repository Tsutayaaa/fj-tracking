# fj-tracking

**English** | [简体中文](README.zh-CN.md)

A [ZhuLab](https://connect.xjtlu.edu.cn/user/zhulab.) project for tracking and quantitative analysis of *Flavobacterium johnsoniae* motility from microscopy videos.

## Project status

The project is in its initial setup stage. Video processing, cell tracking, and quantitative analysis are not implemented yet. The directory structure and dependencies will be added as development progresses.

## Workspace setup

The current entry point uses only the Python standard library and runs on Python 3.9 or later.

```bash
git clone https://github.com/Tsutayaaa/fj-tracking.git
cd fj-tracking
python3 -m venv .venv
```

Activate the environment on macOS / Linux:

```bash
source .venv/bin/activate
```

On Windows (Command Prompt):

```bat
.venv\Scripts\activate.bat
```

In PyCharm, open the repository and select `.venv` as the project interpreter. To check the entry point:

```bash
python main.py
```

It currently prints the project status only.

## Workspace conventions

- `.editorconfig` defines UTF-8 encoding, LF line endings, and four-space indentation for Python.
- `.gitignore` excludes local virtual environments, IDE settings, Python caches, build artifacts, and local environment files.
- Add source folders, data organization, and dependency configuration when they are needed.

## License

No open-source license has been selected yet.
