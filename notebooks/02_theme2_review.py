"""Marimo review app for Theme 2 PCA, clustering and anomalies."""

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
def _(json, pd, root):
    tables = root / "outputs" / "tables"
    summary = json.loads((tables / "theme2_summary.json").read_text(encoding="utf-8"))
    clusters = pd.read_csv(tables / "theme2_cluster_summary.csv")
    daily_anomalies = pd.read_csv(tables / "theme2_daily_anomalies.csv").head(20)
    weekly_anomalies = pd.read_csv(tables / "theme2_weekly_anomalies.csv").head(20)
    return clusters, daily_anomalies, summary, weekly_anomalies


@app.cell
def _(mo, summary):
    mo.md(
        f"""
        # 主题 2：周变化、日变化、PCA 与聚类

        - 完整日：**{summary["complete_days"]:,}**；完整周：**{summary["complete_weeks"]:,}**
        - 达到 90% 累计解释方差所需主成分：**{summary["pca_components_for_90_percent"]}**
        - 自动选择聚类数：**{summary["selected_cluster_count"]}**

        异常分数用于相对排序；外部事件仅作解释线索，不代表因果关系。
        """
    )
    return


@app.cell
def _(clusters):
    clusters  # noqa: B018 - final expression is the marimo cell output
    return


@app.cell
def _(daily_anomalies, mo):
    mo.ui.table(daily_anomalies, label="日异常候选")
    return


@app.cell
def _(mo, weekly_anomalies):
    mo.ui.table(weekly_anomalies, label="周异常候选")
    return


if __name__ == "__main__":
    app.run()
