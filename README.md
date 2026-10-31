# 弗里蒙特桥流量时序分析与预测

本项目分析西雅图 Fremont Bridge 自行车与滑板车小时计数数据，并按课程要求依次完成：

1. 年变化和月变化；
2. 周变化和日变化，包括 PCA 与聚类；
3. 以 2026 年以前数据训练、2026 年数据测试的流量预测。

## 开发顺序

```text
项目与环境 → 数据获取与校验 → 主题1 → 主题2 → 主题3 → 报告与幻灯片 → WSL/Linux复现
```

主题模块在前一阶段通过测试后再进入下一阶段，公共数据处理逻辑放在 `src/`，marimo 分析入口放在 `notebooks/`。

## Windows 环境

项目当前使用 Python 3.12。首次初始化可运行：

```powershell
.\scripts\bootstrap.ps1
```

项目内的包装脚本会把 uv 的 Python 和缓存放在 `tools/`，避免依赖全局安装。后续命令：

```powershell
.\scripts\uv.ps1 sync
.\scripts\uv.ps1 run fremont doctor
.\scripts\uv.ps1 run pytest
```

启动 marimo：

```powershell
.\scripts\uv.ps1 run marimo edit
```

## Linux / WSL 复现

不要复制 Windows 的 `.venv`。克隆仓库后重新创建环境：

```bash
uv sync
uv run fremont doctor
uv run pytest
```

## 数据

原始数据来自 Seattle Open Data 的 `65db-xm6k` 数据集。运行：

```powershell
.\scripts\uv.ps1 run fremont download
```

原始数据保存在 `data/raw/`，默认不提交 Git。下载元数据会同时保存为 JSON，以记录下载日期、URL、字节数和 SHA-256。

## 可选模型环境

基础环境只安装完成主题1、主题2及统计预测所需依赖。进入主题3后按选定路线安装：

```powershell
.\scripts\uv.ps1 sync --extra boosting
.\scripts\uv.ps1 sync --extra deep-learning
```

