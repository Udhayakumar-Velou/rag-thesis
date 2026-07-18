from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Output directory
# =============================================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Display names
# =============================================================================

CHUNKER_NAMES = {
    "FixedChunker": "Fixed",
    "RecursiveChunker": "Recursive",
    "SemanticChunker": "Semantic",
    "HierarchicalChunker": "Hierarchical",
    "AdaptiveChunker": "Adaptive",
}

RETRIEVER_NAMES = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}

COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
]


def save_figure(fig, filename: str):
    png = OUTPUT_DIR / f"{filename}.png"
    pdf = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# =============================================================================
# Simple Bar Chart
# =============================================================================

def plot_bar_metric(
    df: pd.DataFrame,
    group_by: str,
    metric: str,
    title: str,
    filename: str,
):

    summary = (
        df.groupby(group_by)[metric]
        .mean()
        .sort_values(ascending=False)
    )

    if group_by == "Chunker":
        summary.index = [CHUNKER_NAMES.get(x, x) for x in summary.index]

    if group_by == "Retriever":
        summary.index = [
            RETRIEVER_NAMES.get(str(x).lower(), x)
            for x in summary.index
        ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary.index,
        summary.values,
        color=COLORS * 3,
        edgecolor="black",
    )

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(group_by)
    ax.set_ylabel(metric)

    if metric != "Latency(ms)":
        ax.set_ylim(0, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + (0.01 if metric != "Latency(ms)" else h*0.02),
            f"{h:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    save_figure(fig, filename)

    plt.close(fig)


# =============================================================================
# GROUPED BAR CHART (Research Figure)
# =============================================================================

def plot_grouped_metric(
    df: pd.DataFrame,
    metric: str,
    filename: str,
):

    pivot = (
        df.pivot_table(
            index="Chunker",
            columns="Retriever",
            values=metric,
            aggfunc="mean",
        )
        .fillna(0)
    )

    pivot = pivot.rename(index=CHUNKER_NAMES)

    pivot = pivot.rename(
        columns=lambda c: RETRIEVER_NAMES.get(
            str(c).lower(),
            c,
        )
    )

    retrievers = list(pivot.columns)

    chunkers = list(pivot.index)

    x = np.arange(len(chunkers))

    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))

    for i, retriever in enumerate(retrievers):

        values = pivot[retriever].values

        bars = ax.bar(
            x + (i-1)*width,
            values,
            width,
            label=retriever,
            color=COLORS[i],
            edgecolor="black",
        )

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()+bar.get_width()/2,
                h+0.01,
                f"{h:.2f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(chunkers)

    ax.set_ylim(0,1.05)

    ax.set_ylabel(metric)

    ax.set_title(
        f"{metric} by Chunking Strategy and Retrieval Method",
        fontsize=15,
        weight="bold",
    )

    ax.legend(title="Retriever")

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    save_figure(fig, filename)

    plt.close(fig)