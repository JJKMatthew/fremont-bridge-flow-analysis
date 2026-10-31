"""Altair chart styling and deterministic export."""

from __future__ import annotations

from pathlib import Path

import altair as alt

alt.data_transformers.disable_max_rows()


def style_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    return (
        chart.configure(font="Microsoft YaHei")
        .configure_axis(labelFontSize=12, titleFontSize=13, gridOpacity=0.2)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=18, anchor="start", color="#25364a")
        .properties(background="white")
    )


def export_chart(
    chart: alt.Chart | alt.LayerChart,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    styled = style_chart(chart)
    html_path = output_dir / f"{name}.html"
    png_path = output_dir / f"{name}.png"
    styled.save(str(html_path))
    styled.save(str(png_path), scale_factor=2.0)
    return html_path, png_path
