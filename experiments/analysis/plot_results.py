from .heatmaps import plot_heatmap
from .plotting import (
    plot_bar_metric,
    plot_grouped_metric,
)
from .utils import load_results


def main():
    df = load_results()

    metrics = [
        "Recall",
        "Precision",
        "MRR",
        "HitRate",
    ]

    # ==================================================
    # Average Bar Charts
    # ==================================================

    for metric in metrics:
        plot_bar_metric(
            df=df,
            group_by="Chunker",
            metric=metric,
            title=f"Average {metric} by Chunking Strategy",
            filename=f"{metric.lower()}_by_chunker",
        )

        plot_bar_metric(
            df=df,
            group_by="Retriever",
            metric=metric,
            title=f"Average {metric} by Retrieval Method",
            filename=f"{metric.lower()}_by_retriever",
        )

    # ==================================================
    # Latency
    # ==================================================

    plot_bar_metric(
        df=df,
        group_by="Retriever",
        metric="Latency(ms)",
        title="Average Retrieval Latency",
        filename="latency_by_retriever",
    )

    # ==================================================
    # Grouped Comparison Charts
    # ==================================================

    for metric in metrics:
        plot_grouped_metric(
            df=df,
            metric=metric,
            filename=f"grouped_{metric.lower()}",
        )

    # ==================================================
    # Heatmaps (Overall)
    # ==================================================

    for metric in metrics:
        plot_heatmap(
            df=df,
            metric=metric,
            filename=f"{metric.lower()}_heatmap",
        )


if __name__ == "__main__":
    main()