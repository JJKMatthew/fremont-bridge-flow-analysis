import pandas as pd

from fremont_bridge_analysis.theme2.analysis import (
    _annotate_external_events,
    _cluster_label_mapping,
)


def test_cluster_labels_use_weekend_composition() -> None:
    days = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1, 1],
            "is_weekend": [True, True, False, False, False, False],
        }
    )

    assert _cluster_label_mapping(days) == {0: "周末/休闲型", 1: "工作日通勤型"}


def test_external_event_annotation_matches_week_overlap() -> None:
    weeks = pd.DataFrame({"week_start": pd.to_datetime(["2022-12-19", "2023-01-09"])})
    events = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2022-12-22"]),
            "end_date": pd.to_datetime(["2022-12-23"]),
            "event": ["冻雨"],
            "category": ["冬季天气"],
            "source_url": ["https://example.test/event"],
        }
    )

    annotated = _annotate_external_events(weeks, "week_start", events, span_days=6)

    assert annotated.loc[0, "external_event"] == "冻雨"
    assert annotated.loc[1, "external_event"] == ""
