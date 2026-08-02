"""Publication-quality visualization for research evaluation results."""

from deep_research.visualization.charts import (
    dimension_heatmap,
    bootstrap_ci_plot,
    radar_chart,
    cost_quality_scatter,
    critical_difference_diagram,
    ablation_bar_chart,
    concordance_heatmap,
    performance_profile,
    generate_all_figures,
    PATTERN_NAMES,
    PATTERN_COLORS,
    DIMENSION_DISPLAY,
)

__all__ = [
    "dimension_heatmap",
    "bootstrap_ci_plot",
    "radar_chart",
    "cost_quality_scatter",
    "critical_difference_diagram",
    "ablation_bar_chart",
    "concordance_heatmap",
    "performance_profile",
    "generate_all_figures",
    "PATTERN_NAMES",
    "PATTERN_COLORS",
    "DIMENSION_DISPLAY",
]
