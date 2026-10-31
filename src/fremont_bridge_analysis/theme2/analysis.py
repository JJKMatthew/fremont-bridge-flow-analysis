"""Theme 2: weekly/daily patterns, PCA, clustering and anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import altair as alt
import holidays
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from fremont_bridge_analysis.analysis_utils import robust_zscore, write_json
from fremont_bridge_analysis.config import ProjectConfig
from fremont_bridge_analysis.visualization.charts import export_chart

WEEKDAY_NAMES = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
    5: "周六",
    6: "周日",
}


@dataclass(frozen=True)
class Theme2Result:
    summary_path: Path
    cluster_path: Path
    daily_anomaly_path: Path
    weekly_anomaly_path: Path


def _load_hourly(config: ProjectConfig) -> pd.DataFrame:
    path = config.path("processed") / "fremont_hourly.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed data not found: {path}. Run `fremont prepare` first.")
    frame = pd.read_parquet(path)
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _holiday_mapping(start_year: int, end_year: int) -> dict[pd.Timestamp, str]:
    calendar = holidays.US(years=range(start_year, end_year + 1), subdiv="WA")
    return {pd.Timestamp(day): str(name) for day, name in calendar.items()}


def _load_external_events(config: ProjectConfig) -> pd.DataFrame:
    path = config.root / "config" / "external_events.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["start_date", "end_date", "event", "category", "source_url"])
    events = pd.read_csv(path)
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    return events


def _annotate_external_events(
    frame: pd.DataFrame,
    date_column: str,
    events: pd.DataFrame,
    span_days: int = 0,
) -> pd.DataFrame:
    annotated = frame.copy()
    names: list[str] = []
    categories: list[str] = []
    sources: list[str] = []
    for value in pd.to_datetime(annotated[date_column]):
        interval_end = value + pd.Timedelta(days=span_days)
        matches = events[events["start_date"].le(interval_end) & events["end_date"].ge(value)]
        names.append("; ".join(matches["event"].astype(str)))
        categories.append("; ".join(matches["category"].astype(str)))
        sources.append("; ".join(matches["source_url"].astype(str)))
    annotated["external_event"] = names
    annotated["external_event_category"] = categories
    annotated["external_event_source"] = sources
    return annotated


def _cluster_label_mapping(cluster_days: pd.DataFrame) -> dict[int, str]:
    weekend_share = cluster_days.groupby("cluster")["is_weekend"].mean()
    mapping: dict[int, str] = {}
    used: set[str] = set()
    for cluster, share in weekend_share.items():
        if share >= 0.60:
            label = "周末/休闲型"
        elif share <= 0.20:
            label = "工作日通勤型"
        else:
            label = "混合型"
        if label in used:
            label = f"{label}（类别{int(cluster) + 1}）"
        mapping[int(cluster)] = label
        used.add(label)
    return mapping


def run_theme2(config: ProjectConfig) -> Theme2Result:
    hourly = _load_hourly(config)
    tables_dir = config.path("tables")
    figures_dir = config.path("figures") / "theme2"
    random_seed = int(config.values["project"]["random_seed"])
    external_events = _load_external_events(config)

    daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            total_flow=("total", lambda values: values.sum(min_count=1)),
            west_flow=("west_sidewalk", lambda values: values.sum(min_count=1)),
            east_flow=("east_sidewalk", lambda values: values.sum(min_count=1)),
            row_count=("timestamp_local", "size"),
            valid_hours=("total", "count"),
        )
        .sort_values("date")
    )
    daily["weekday"] = daily["date"].dt.weekday
    daily["weekday_name"] = daily["weekday"].map(WEEKDAY_NAMES)
    daily["is_weekend"] = daily["weekday"].ge(5)
    daily["is_complete_day"] = daily["row_count"].eq(24) & daily["valid_hours"].eq(24)
    holiday_map = _holiday_mapping(
        int(daily["date"].dt.year.min()), int(daily["date"].dt.year.max())
    )
    daily["holiday"] = daily["date"].map(holiday_map).fillna("")
    daily["is_holiday"] = daily["holiday"].ne("")
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["weekday"], unit="D")
    daily.to_csv(tables_dir / "theme2_daily_summary.csv", index=False, encoding="utf-8-sig")

    complete_daily = daily[daily["is_complete_day"]].copy()
    weekday_summary = (
        complete_daily.groupby(["weekday", "weekday_name"], as_index=False)
        .agg(
            mean_daily_flow=("total_flow", "mean"),
            median_daily_flow=("total_flow", "median"),
            q25_daily_flow=("total_flow", lambda values: values.quantile(0.25)),
            q75_daily_flow=("total_flow", lambda values: values.quantile(0.75)),
            days=("date", "count"),
        )
        .sort_values("weekday")
    )
    weekday_summary.to_csv(
        tables_dir / "theme2_weekday_summary.csv", index=False, encoding="utf-8-sig"
    )

    hourly_valid = hourly.dropna(subset=["total"]).copy()
    hour_weekday = (
        hourly_valid.groupby(["weekday_iso", "hour"], as_index=False)
        .agg(mean_flow=("total", "mean"), median_flow=("total", "median"))
        .rename(columns={"weekday_iso": "weekday"})
    )
    hour_weekday["weekday"] = hour_weekday["weekday"] - 1
    hour_weekday["weekday_name"] = hour_weekday["weekday"].map(WEEKDAY_NAMES)
    hour_weekday.to_csv(
        tables_dir / "theme2_weekday_hour_profile.csv", index=False, encoding="utf-8-sig"
    )

    weekly_profiles = complete_daily.pivot(
        index="week_start", columns="weekday", values="total_flow"
    ).dropna()
    weekly_profiles = weekly_profiles.reindex(columns=range(7))
    weekly_totals = weekly_profiles.sum(axis=1)
    weekly_shares = weekly_profiles.div(weekly_totals, axis=0)
    reference_week = weekly_shares.median(axis=0)
    profile_distance = np.sqrt(((weekly_shares - reference_week) ** 2).sum(axis=1))
    rolling_log_median = np.log1p(weekly_totals).rolling(13, center=True, min_periods=5).median()
    volume_residual = np.log1p(weekly_totals) - rolling_log_median
    weekly = pd.DataFrame(
        {
            "week_start": weekly_profiles.index,
            "weekly_total": weekly_totals,
            "profile_distance": profile_distance,
            "volume_residual": volume_residual,
        }
    ).reset_index(drop=True)
    weekly["profile_robust_z"] = robust_zscore(weekly["profile_distance"].to_numpy())
    weekly["volume_robust_z"] = robust_zscore(weekly["volume_residual"].to_numpy())
    weekly["anomaly_score"] = np.maximum(
        np.abs(weekly["profile_robust_z"]), np.abs(weekly["volume_robust_z"])
    )
    holidays_by_week = (
        complete_daily[complete_daily["is_holiday"]]
        .groupby("week_start")["holiday"]
        .agg(lambda values: "; ".join(sorted(set(values))))
    )
    weekly["holiday"] = weekly["week_start"].map(holidays_by_week).fillna("")
    weekly["contains_holiday"] = weekly["holiday"].ne("")
    weekly = weekly.sort_values("week_start")
    weekly = _annotate_external_events(weekly, "week_start", external_events, span_days=6)
    weekly.to_csv(tables_dir / "theme2_weekly_scores.csv", index=False, encoding="utf-8-sig")
    weekly_anomalies = weekly.sort_values("anomaly_score", ascending=False).head(30).copy()
    weekly_anomaly_path = tables_dir / "theme2_weekly_anomalies.csv"
    weekly_anomalies.to_csv(weekly_anomaly_path, index=False, encoding="utf-8-sig")

    hourly_for_profile = hourly.dropna(subset=["total"])
    profiles = hourly_for_profile.pivot_table(
        index="date", columns="hour", values="total", aggfunc="first"
    ).reindex(columns=range(24))
    row_counts = hourly.groupby("date").size()
    valid_counts = hourly.groupby("date")["total"].count()
    valid_dates = row_counts.index[(row_counts == 24) & (valid_counts == 24)]
    profiles = profiles.loc[profiles.index.intersection(valid_dates)].dropna()
    positive_mask = profiles.sum(axis=1).gt(0)
    profiles = profiles.loc[positive_mask]
    normalized_profiles = profiles.div(profiles.sum(axis=1), axis=0)

    full_pca = PCA(random_state=random_seed)
    full_scores = full_pca.fit_transform(normalized_profiles.to_numpy())
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    component_count = max(2, int(np.searchsorted(cumulative_variance, 0.90) + 1))
    pca_scores = full_scores[:, :component_count]

    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(full_pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": full_pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative_variance,
            "selected_for_clustering": np.arange(1, len(cumulative_variance) + 1)
            <= component_count,
        }
    )
    explained.to_csv(
        tables_dir / "theme2_pca_explained_variance.csv", index=False, encoding="utf-8-sig"
    )
    loadings = pd.DataFrame(
        full_pca.components_[:component_count].T,
        index=range(24),
        columns=[f"PC{index}" for index in range(1, component_count + 1)],
    ).reset_index(names="hour")
    loadings.to_csv(tables_dir / "theme2_pca_loadings.csv", index=False, encoding="utf-8-sig")

    selection_rows: list[dict[str, float | int]] = []
    candidate_models: dict[int, KMeans] = {}
    for cluster_count in range(2, 9):
        model = KMeans(n_clusters=cluster_count, n_init=30, random_state=random_seed)
        labels = model.fit_predict(pca_scores)
        comparison = KMeans(
            n_clusters=cluster_count, n_init=30, random_state=random_seed + 1
        ).fit_predict(pca_scores)
        counts = np.bincount(labels)
        selection_rows.append(
            {
                "cluster_count": cluster_count,
                "silhouette_score": float(silhouette_score(pca_scores, labels)),
                "stability_adjusted_rand": float(adjusted_rand_score(labels, comparison)),
                "minimum_cluster_share": float(counts.min() / counts.sum()),
                "inertia": float(model.inertia_),
            }
        )
        candidate_models[cluster_count] = model
    cluster_selection = pd.DataFrame(selection_rows)
    valid_candidates = cluster_selection[cluster_selection["minimum_cluster_share"].ge(0.03)]
    if valid_candidates.empty:
        valid_candidates = cluster_selection
    selected_k = int(
        valid_candidates.sort_values(
            ["silhouette_score", "stability_adjusted_rand"], ascending=False
        ).iloc[0]["cluster_count"]
    )
    cluster_selection["selected"] = cluster_selection["cluster_count"].eq(selected_k)
    cluster_selection.to_csv(
        tables_dir / "theme2_cluster_selection.csv", index=False, encoding="utf-8-sig"
    )

    cluster_model = candidate_models[selected_k]
    labels = cluster_model.labels_
    selected_reconstruction = (
        full_scores[:, :component_count] @ full_pca.components_[:component_count] + full_pca.mean_
    )
    reconstruction_error = np.mean(
        (normalized_profiles.to_numpy() - selected_reconstruction) ** 2, axis=1
    )
    centroid_distance = np.linalg.norm(pca_scores - cluster_model.cluster_centers_[labels], axis=1)
    distance_z = np.zeros(len(labels), dtype=float)
    for cluster in range(selected_k):
        mask = labels == cluster
        distance_z[mask] = robust_zscore(centroid_distance[mask])
    reconstruction_z = robust_zscore(reconstruction_error)

    cluster_days = pd.DataFrame(
        {
            "date": pd.to_datetime(normalized_profiles.index),
            "cluster": labels,
            "centroid_distance": centroid_distance,
            "distance_robust_z": distance_z,
            "reconstruction_error": reconstruction_error,
            "reconstruction_robust_z": reconstruction_z,
        }
    )
    for index in range(min(3, component_count)):
        cluster_days[f"PC{index + 1}"] = pca_scores[:, index]
    cluster_days["anomaly_score"] = np.maximum(
        np.abs(cluster_days["distance_robust_z"]),
        np.abs(cluster_days["reconstruction_robust_z"]),
    )
    cluster_days = cluster_days.merge(
        daily[
            [
                "date",
                "total_flow",
                "weekday",
                "weekday_name",
                "is_weekend",
                "holiday",
                "is_holiday",
            ]
        ],
        on="date",
        how="left",
    )
    label_mapping = _cluster_label_mapping(cluster_days)
    cluster_days["cluster_label"] = cluster_days["cluster"].map(label_mapping)
    cluster_path = tables_dir / "theme2_daily_clusters.csv"
    cluster_days.to_csv(cluster_path, index=False, encoding="utf-8-sig")

    cluster_summary = (
        cluster_days.groupby(["cluster", "cluster_label"], as_index=False)
        .agg(
            days=("date", "count"),
            mean_daily_flow=("total_flow", "mean"),
            median_daily_flow=("total_flow", "median"),
            weekend_share=("is_weekend", "mean"),
            holiday_share=("is_holiday", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
        )
        .sort_values("cluster")
    )
    cluster_summary.to_csv(
        tables_dir / "theme2_cluster_summary.csv", index=False, encoding="utf-8-sig"
    )

    cluster_profile_rows: list[dict[str, float | int | str]] = []
    for cluster in range(selected_k):
        mask = labels == cluster
        mean_profile = normalized_profiles.to_numpy()[mask].mean(axis=0)
        for hour, share in enumerate(mean_profile):
            cluster_profile_rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": label_mapping[cluster],
                    "hour": hour,
                    "share": share,
                }
            )
    cluster_profiles = pd.DataFrame(cluster_profile_rows)
    cluster_profiles.to_csv(
        tables_dir / "theme2_cluster_profiles.csv", index=False, encoding="utf-8-sig"
    )

    weekday_mix = (
        cluster_days.groupby(["cluster_label", "weekday", "weekday_name"], as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )
    weekday_mix["share_within_cluster"] = weekday_mix["days"] / weekday_mix.groupby(
        "cluster_label"
    )["days"].transform("sum")
    weekday_mix.to_csv(
        tables_dir / "theme2_cluster_weekday_mix.csv", index=False, encoding="utf-8-sig"
    )

    daily_anomalies = cluster_days.sort_values("anomaly_score", ascending=False).head(40).copy()
    daily_anomalies = _annotate_external_events(
        daily_anomalies, "date", external_events, span_days=0
    )
    daily_anomaly_path = tables_dir / "theme2_daily_anomalies.csv"
    daily_anomalies.to_csv(daily_anomaly_path, index=False, encoding="utf-8-sig")

    complete_daily["day_type"] = np.where(complete_daily["is_weekend"], "周末", "工作日")
    cluster_days["day_type"] = np.where(cluster_days["is_weekend"], "周末", "工作日")
    weekday_chart = (
        alt.Chart(complete_daily)
        .mark_boxplot(size=45, extent=1.5)
        .encode(
            x=alt.X(
                "weekday_name:N",
                title="星期",
                sort=[WEEKDAY_NAMES[index] for index in range(7)],
            ),
            y=alt.Y("total_flow:Q", title="日流量"),
            color=alt.Color("day_type:N", title="日期类型"),
        )
        .properties(width=900, height=430, title="一周七天流量分布")
    )
    export_chart(weekday_chart, figures_dir, "weekday_distribution")

    heatmap = (
        alt.Chart(hour_weekday)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="小时"),
            y=alt.Y(
                "weekday_name:N",
                title="星期",
                sort=[WEEKDAY_NAMES[index] for index in range(7)],
            ),
            color=alt.Color("mean_flow:Q", title="平均流量", scale=alt.Scale(scheme="blues")),
            tooltip=["weekday_name", "hour", "mean_flow", "median_flow"],
        )
        .properties(width=900, height=430, title="星期与小时流量模式")
    )
    export_chart(heatmap, figures_dir, "weekday_hour_heatmap")

    weekly_chart = (
        alt.Chart(weekly)
        .mark_line(color="#2b7fb8")
        .encode(
            x=alt.X("week_start:T", title="周开始日期"),
            y=alt.Y("anomaly_score:Q", title="异常分数"),
            tooltip=[
                "week_start:T",
                "weekly_total",
                "anomaly_score",
                "holiday",
                "external_event",
            ],
        )
        .properties(width=900, height=430, title="异常周检测")
    )
    export_chart(weekly_chart, figures_dir, "weekly_anomaly_timeline")

    pca_chart = (
        alt.Chart(cluster_days)
        .mark_circle(size=45, opacity=0.65)
        .encode(
            x=alt.X("PC1:Q", title="第一主成分"),
            y=alt.Y("PC2:Q", title="第二主成分"),
            color=alt.Color("cluster_label:N", title="日模式类别"),
            shape=alt.Shape("day_type:N", title="日期类型"),
            tooltip=["date:T", "cluster_label", "weekday_name", "total_flow", "holiday"],
        )
        .properties(width=900, height=500, title="日内流量曲线的PCA与聚类")
    )
    export_chart(pca_chart, figures_dir, "pca_cluster_scatter")

    profile_chart = (
        alt.Chart(cluster_profiles)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("hour:Q", title="小时", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("share:Q", title="日流量占比"),
            color=alt.Color("cluster_label:N", title="日模式类别"),
            tooltip=["cluster_label", "hour", "share"],
        )
        .properties(width=900, height=430, title="各类别24小时流量形状")
    )
    export_chart(profile_chart, figures_dir, "cluster_profiles")

    mix_chart = (
        alt.Chart(weekday_mix)
        .mark_rect()
        .encode(
            x=alt.X(
                "weekday_name:N",
                title="星期",
                sort=[WEEKDAY_NAMES[index] for index in range(7)],
            ),
            y=alt.Y("cluster_label:N", title="日模式类别"),
            color=alt.Color(
                "share_within_cluster:Q", title="类别内占比", scale=alt.Scale(scheme="oranges")
            ),
            tooltip=["cluster_label", "weekday_name", "days", "share_within_cluster"],
        )
        .properties(width=900, height=300, title="聚类类别的星期构成")
    )
    export_chart(mix_chart, figures_dir, "cluster_weekday_mix")

    summary_path = tables_dir / "theme2_summary.json"
    write_json(
        summary_path,
        {
            "complete_days": len(complete_daily),
            "incomplete_days": int((~daily["is_complete_day"]).sum()),
            "complete_weeks": len(weekly),
            "profile_days_used_for_pca": len(normalized_profiles),
            "pca_components_for_90_percent": component_count,
            "pca_cumulative_variance": cumulative_variance[component_count - 1],
            "selected_cluster_count": selected_k,
            "selected_cluster_metrics": cluster_selection.loc[
                cluster_selection["selected"]
            ].to_dict(orient="records")[0],
            "cluster_summary": cluster_summary.to_dict(orient="records"),
            "top_weekly_anomalies": weekly_anomalies.head(10).to_dict(orient="records"),
            "top_daily_anomalies": daily_anomalies.head(15).to_dict(orient="records"),
            "method_notes": {
                "profile_normalization": "Each 24-hour vector is divided by its daily total.",
                "pca_selection": "Minimum components reaching 90% cumulative explained variance.",
                "cluster_selection": (
                    "Highest silhouette score among solutions whose smallest cluster contains at "
                    "least 3% of days; stability is reported using adjusted Rand index."
                ),
                "anomaly_score": (
                    "Maximum absolute robust z-score of within-cluster centroid distance and PCA "
                    "reconstruction error; use it as a relative ranking, not a probability."
                ),
                "external_events": (
                    "Official Seattle/King County event windows are joined by date overlap for "
                    "interpretation only; matches do not establish causality."
                ),
            },
        },
    )
    return Theme2Result(
        summary_path=summary_path,
        cluster_path=cluster_path,
        daily_anomaly_path=daily_anomaly_path,
        weekly_anomaly_path=weekly_anomaly_path,
    )
