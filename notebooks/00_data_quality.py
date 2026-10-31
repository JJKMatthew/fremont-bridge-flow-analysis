"""Marimo review app for the prepared dataset and quality checks."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, json, mo, pd


@app.cell
def _(Path):
    root = Path(__file__).resolve().parents[1]
    return (root,)


@app.cell
def _(json, root):
    quality_path = root / "outputs" / "tables" / "data_quality_summary.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    return quality, quality_path


@app.cell
def _(mo, quality, quality_path):
    mo.md(
        f"""
        # 数据质量检查

        结果文件：`{quality_path}`

        - 原始行数：**{quality["rows"]:,}**
        - 重复时间戳：**{quality["duplicate_timestamps"]:,}**
        - 缺失小时：**{quality["absent_hourly_timestamps"]:,}**
        - 含缺失计数的行：**{quality["rows_with_missing_counts"]:,}**
        """
    )
    return


@app.cell
def _(pd, quality):
    pd.DataFrame(quality["annual_coverage"])
    return


if __name__ == "__main__":
    app.run()
