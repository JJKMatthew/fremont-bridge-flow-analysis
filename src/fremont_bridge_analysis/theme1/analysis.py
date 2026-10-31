"""Theme 1: annual/monthly patterns, anomalies and sidewalk comparison."""

from __future__ import annotations

from calendar import isleap
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import TheilSenRegressor

from fremont_bridge_analysis.analysis_utils import robust_zscore, write_json
from fremont_bridge_analysis.config import ProjectConfig
from fremont_bridge_analysis.visualization.charts import export_chart


@dataclass(frozen=True)
class Theme1Result:
    summary_path: Path
    annual_path: Path
    monthly_path: Path


def _expected_hours(year: int) -> int:
    return 8784 if isleap(year) else 8760


def _load_hourly(config: ProjectConfig) -> pd.DataFrame:
    path = config.path("processed") / "fremont_hourly.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed data not found: {path}. Run `fremont prepare` first.")
    frame = pd.read_parquet(path)
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _theil_sen_slope(frame: pd.DataFrame, random_seed: int) -> float:
    model = TheilSenRegressor(random_state=random_seed)
    model.fit(frame[["year"]], frame["mean_daily_flow"])
    return float(model.coef_[0])


def run_theme1(config: ProjectConfig) -> Theme1Result:
    hourly = _load_hourly(config)
    tables_dir = config.path("tables")
    figures_dir = config.path("figures") / "theme1"
    tables_dir.mkdir(parents=True, exist_ok=True)

    daily = (
        hourly.groupby(["year", "date"], as_index=False)
        .agg(
            total_flow=("total", lambda values: values.sum(min_count=1)),
            west_flow=("west_sidewalk", lambda values: values.sum(min_count=1)),
            east_flow=("east_sidewalk", lambda values: values.sum(min_count=1)),
            row_count=("timestamp_local", "size"),
            valid_hours=("total", "count"),
        )
        .sort_values("date")
    )

    annual = (
        hourly.groupby("year", as_index=False)
        .agg(
            total_flow=("total", lambda values: values.sum(min_count=1)),
            west_flow=("west_sidewalk", lambda values: values.sum(min_count=1)),
            east_flow=("east_sidewalk", lambda values: values.sum(min_count=1)),
            mean_hourly_flow=("total", "mean"),
            row_count=("timestamp_local", "size"),
            valid_hours=("total", "count"),
            first_timestamp=("timestamp_local", "min"),
            last_timestamp=("timestamp_local", "max"),
        )
        .sort_values("year")
    )
    daily_means = daily.groupby("year", as_index=False).agg(
        mean_daily_flow=("total_flow", "mean"),
        median_daily_flow=("total_flow", "median"),
        valid_days=("total_flow", "count"),
    )
    annual = annual.merge(daily_means, on="year", how="left")
    annual["expected_full_year_hours"] = annual["year"].map(_expected_hours)
    annual["row_coverage"] = annual["row_count"] / annual["expected_full_year_hours"]
    annual["value_coverage"] = annual["valid_hours"] / annual["expected_full_year_hours"]
    annual["is_complete_year"] = annual["row_coverage"].ge(0.999)

    trend_data = annual.loc[annual["is_complete_year"] & annual["year"].between(2013, 2025)].copy()
    random_seed = int(config.values["project"]["random_seed"])
    trend_model = TheilSenRegressor(random_state=random_seed)
    trend_model.fit(trend_data[["year"]], trend_data["mean_daily_flow"])
    annual["trend_mean_daily_flow"] = trend_model.predict(annual[["year"]])
    annual["trend_residual"] = annual["mean_daily_flow"] - annual["trend_mean_daily_flow"]
    complete_mask = annual["is_complete_year"] & annual["year"].between(2013, 2025)
    annual["anomaly_robust_z"] = np.nan
    annual.loc[complete_mask, "anomaly_robust_z"] = robust_zscore(
        annual.loc[complete_mask, "trend_residual"].to_numpy()
    )
    # This is a screening flag for historical interpretation, not a causal claim.
    annual["is_anomaly_candidate"] = annual["anomaly_robust_z"].abs().ge(1.75)
    annual_path = tables_dir / "theme1_annual_summary.csv"
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")

    month_by_year = (
        hourly.groupby(["year", "month"], as_index=False)
        .agg(
            total_flow=("total", lambda values: values.sum(min_count=1)),
            mean_hourly_flow=("total", "mean"),
            valid_hours=("total", "count"),
        )
        .sort_values(["year", "month"])
    )
    complete_months = month_by_year[month_by_year["year"].between(2013, 2025)].copy()
    monthly = (
        complete_months.groupby("month", as_index=False)
        .agg(
            mean_hourly_flow=("mean_hourly_flow", "mean"),
            median_hourly_flow=("mean_hourly_flow", "median"),
            q25_hourly_flow=("mean_hourly_flow", lambda values: values.quantile(0.25)),
            q75_hourly_flow=("mean_hourly_flow", lambda values: values.quantile(0.75)),
            mean_month_total=("total_flow", "mean"),
        )
        .sort_values("month")
    )
    monthly["seasonal_index"] = monthly["mean_hourly_flow"] / monthly["mean_hourly_flow"].mean()
    monthly_path = tables_dir / "theme1_monthly_summary.csv"
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    month_by_year.to_csv(
        tables_dir / "theme1_year_month_summary.csv", index=False, encoding="utf-8-sig"
    )

    daily_sides = daily.dropna(subset=["west_flow", "east_flow"]).copy()
    side_difference = daily_sides["west_flow"] - daily_sides["east_flow"]
    paired_t = stats.ttest_rel(daily_sides["west_flow"], daily_sides["east_flow"])
    wilcoxon = stats.wilcoxon(side_difference)
    effect_size = float(side_difference.mean() / side_difference.std(ddof=1))

    hourly_profile = (
        hourly.groupby("hour", as_index=False)
        .agg(
            west_mean=("west_sidewalk", "mean"),
            east_mean=("east_sidewalk", "mean"),
            total_mean=("total", "mean"),
        )
        .sort_values("hour")
    )
    hourly_profile.to_csv(
        tables_dir / "theme1_sidewalk_hour_profile.csv", index=False, encoding="utf-8-sig"
    )

    annual_long = annual.loc[complete_mask].melt(
        id_vars=["year", "is_complete_year"],
        value_vars=["west_flow", "east_flow"],
        var_name="sidewalk",
        value_name="flow",
    )
    annual_long["sidewalk"] = annual_long["sidewalk"].map(
        {"west_flow": "西侧人行道", "east_flow": "东侧人行道"}
    )
    annual["coverage_label"] = np.where(annual["is_complete_year"], "完整年度", "不完整年度")
    complete_line = (
        alt.Chart(annual.loc[complete_mask])
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("year:O", title="年份"),
            y=alt.Y("mean_daily_flow:Q", title="日均流量"),
            tooltip=["year", "total_flow", "mean_daily_flow", "value_coverage"],
        )
    )
    trend_line = (
        alt.Chart(annual.loc[complete_mask])
        .mark_line(strokeDash=[7, 5], color="#606060", strokeWidth=2)
        .encode(x="year:O", y="trend_mean_daily_flow:Q")
    )
    partial_points = (
        alt.Chart(annual.loc[~annual["is_complete_year"]])
        .mark_point(filled=True, size=95, color="#9a9a9a")
        .encode(
            x="year:O",
            y="mean_daily_flow:Q",
            tooltip=["year", "mean_daily_flow", "value_coverage", "coverage_label"],
        )
    )
    trend_chart = (complete_line + trend_line + partial_points).properties(
        width=900,
        height=430,
        title="年度日均流量变化（灰点为不完整年度，不纳入趋势拟合）",
    )
    export_chart(trend_chart, figures_dir, "annual_daily_mean_trend")

    monthly_band = (
        alt.Chart(monthly)
        .mark_area(opacity=0.22, color="#2b7fb8")
        .encode(
            x=alt.X("month:O", title="月份"),
            y=alt.Y("q25_hourly_flow:Q", title="小时平均流量"),
            y2="q75_hourly_flow:Q",
        )
    )
    monthly_line = (
        alt.Chart(monthly)
        .mark_line(point=True, strokeWidth=3, color="#c1361c")
        .encode(
            x=alt.X("month:O", title="月份"),
            y=alt.Y("median_hourly_flow:Q", title="小时平均流量"),
            tooltip=["month", "mean_hourly_flow", "median_hourly_flow", "seasonal_index"],
        )
    )
    export_chart(
        (monthly_band + monthly_line).properties(width=900, height=430, title="月度季节分布"),
        figures_dir,
        "monthly_seasonality",
    )

    heatmap = (
        alt.Chart(month_by_year)
        .mark_rect()
        .encode(
            x=alt.X("month:O", title="月份"),
            y=alt.Y("year:O", title="年份"),
            color=alt.Color(
                "mean_hourly_flow:Q", title="小时平均流量", scale=alt.Scale(scheme="blues")
            ),
            tooltip=["year", "month", "total_flow", "mean_hourly_flow", "valid_hours"],
        )
        .properties(width=900, height=430, title="年度与月份流量热力图")
    )
    export_chart(heatmap, figures_dir, "year_month_heatmap")

    side_chart = (
        alt.Chart(annual_long)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("year:O", title="年份"),
            y=alt.Y("flow:Q", title="年度流量"),
            color=alt.Color("sidewalk:N", title="桥面侧别"),
            tooltip=["year", "sidewalk", "flow"],
        )
        .properties(width=900, height=430, title="东西侧人行道年度流量")
    )
    export_chart(side_chart, figures_dir, "annual_sidewalk_comparison")

    hourly_long = hourly_profile.melt(
        id_vars="hour",
        value_vars=["west_mean", "east_mean"],
        var_name="sidewalk",
        value_name="mean_flow",
    )
    hourly_long["sidewalk"] = hourly_long["sidewalk"].map(
        {"west_mean": "西侧人行道", "east_mean": "东侧人行道"}
    )
    hour_chart = (
        alt.Chart(hourly_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("hour:Q", title="小时", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("mean_flow:Q", title="平均流量"),
            color=alt.Color("sidewalk:N", title="桥面侧别"),
            tooltip=["hour", "sidewalk", "mean_flow"],
        )
        .properties(width=900, height=430, title="东西侧人行道24小时平均流量")
    )
    export_chart(hour_chart, figures_dir, "sidewalk_hour_profile")

    anomalous_years = annual.loc[
        annual["is_anomaly_candidate"],
        [
            "year",
            "mean_daily_flow",
            "trend_residual",
            "anomaly_robust_z",
        ],
    ]
    anomalous_years.to_csv(
        tables_dir / "theme1_anomaly_candidates.csv", index=False, encoding="utf-8-sig"
    )

    first_complete = trend_data.iloc[0]
    last_complete = trend_data.iloc[-1]
    pre_covid = trend_data[trend_data["year"].between(2013, 2019)]
    recovery = trend_data[trend_data["year"].between(2021, 2025)]
    summary_path = tables_dir / "theme1_summary.json"
    write_json(
        summary_path,
        {
            "analysis_period": {
                "raw_start": hourly["timestamp_local"].min(),
                "raw_end": hourly["timestamp_local"].max(),
                "complete_year_comparison": [
                    int(first_complete["year"]),
                    int(last_complete["year"]),
                ],
            },
            "long_term_change": {
                "first_complete_year_mean_daily": first_complete["mean_daily_flow"],
                "last_complete_year_mean_daily": last_complete["mean_daily_flow"],
                "percentage_change": 100
                * (last_complete["mean_daily_flow"] / first_complete["mean_daily_flow"] - 1),
                "theil_sen_daily_flow_per_year": float(trend_model.coef_[0]),
                "pre_covid_2013_2019_slope": _theil_sen_slope(pre_covid, random_seed),
                "recovery_2021_2025_slope": _theil_sen_slope(recovery, random_seed),
                "interpretation": (
                    "The full-period slope crosses the COVID-era structural break. Compare it "
                    "with the pre-COVID and recovery segment slopes; do not describe the series "
                    "as one uninterrupted linear trend."
                ),
            },
            "highest_month": int(monthly.loc[monthly["mean_hourly_flow"].idxmax(), "month"]),
            "lowest_month": int(monthly.loc[monthly["mean_hourly_flow"].idxmin(), "month"]),
            "anomaly_candidates": anomalous_years.to_dict(orient="records"),
            "anomaly_note": (
                "Candidates use |robust z| >= 1.75 on full-period trend residuals and require "
                "historical interpretation; this is screening, not proof of causality."
            ),
            "sidewalk_comparison": {
                "west_daily_mean": daily_sides["west_flow"].mean(),
                "east_daily_mean": daily_sides["east_flow"].mean(),
                "mean_paired_difference_west_minus_east": side_difference.mean(),
                "paired_t_statistic": paired_t.statistic,
                "paired_t_pvalue": paired_t.pvalue,
                "wilcoxon_statistic": wilcoxon.statistic,
                "wilcoxon_pvalue": wilcoxon.pvalue,
                "paired_effect_size_dz": effect_size,
                "interpretation_warning": (
                    "The source columns identify west/east sidewalks. They should not be renamed "
                    "northbound/southbound without authoritative metadata."
                ),
            },
        },
    )
    return Theme1Result(
        summary_path=summary_path, annual_path=annual_path, monthly_path=monthly_path
    )
