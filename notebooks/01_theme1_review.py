"""Marimo review app for Theme 1 annual/monthly results."""

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
    summary = json.loads((tables / "theme1_summary.json").read_text(encoding="utf-8"))
    annual = pd.read_csv(tables / "theme1_annual_summary.csv")
    monthly = pd.read_csv(tables / "theme1_monthly_summary.csv")
    return annual, monthly, summary


@app.cell
def _(mo, summary):
    change = summary["long_term_change"]
    mo.md(
        f"""
        # 主题 1：年变化与月变化

        2013–2025 完整年度的首末日均流量变化：**{change["percentage_change"]:.1f}%**。

        - 全期 Theil–Sen 斜率：{change["theil_sen_daily_flow_per_year"]:.1f} 人次/日/年
        - 疫情前斜率（2013–2019）：{change["pre_covid_2013_2019_slope"]:.1f}
        - 恢复期斜率（2021–2025）：{change["recovery_2021_2025_slope"]:.1f}
        - 最高月份：{summary["highest_month"]} 月；最低月份：{summary["lowest_month"]} 月

        全期斜率跨越结构突变，不能单独解释为持续下降趋势。
        """
    )
    return


@app.cell
def _(annual):
    annual  # noqa: B018 - final expression is the marimo cell output
    return


@app.cell
def _(monthly):
    monthly  # noqa: B018 - final expression is the marimo cell output
    return


if __name__ == "__main__":
    app.run()
